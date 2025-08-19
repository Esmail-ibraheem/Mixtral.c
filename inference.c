// mixtral.c — Minimal Mixtral‑style MoE Transformer in pure C (single file)
//
// What this is:
//   • A tiny, readable reference that runs an autoregressive forward pass with:
//       - token embeddings
//       - single transformer layer (pre‑norm)
//       - multi‑head self‑attention with RoPE and a KV cache
//       - MoE feed‑forward with top‑2 gating (SwiGLU)
//       - final LM head to produce logits over a small vocab
//   • Pure C (C99). No dependencies except libm.
//
// What this is NOT:
//   • A full Mixtral 8×7B loader. Shapes here are small for demo; weight I/O is simplified.
//   • Numerically identical to real Mixtral (e.g., no GQA, no exact init, no quantization).
//
// How to build:
//   gcc -O3 -march=native mixtral.c -o mixtral -lm
//
// How to run (toy demo with random weights):
//   ./mixtral "1 5 4 2 3"   # space‑separated token ids from 0..VOCAB-1
//
// Extend to real models:
//   • Increase dims and layer count;
//   • Implement weight loaders (e.g., per‑tensor .bin floats); 
//   • Switch to GQA, add multi‑layer stack; 
//   • Map shapes to Mixtral configs; 
//   • Replace random init with actual weights.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ======= Tiny model config (adjust as you like) =======
#define VOCAB      1000
#define MAX_SEQ     128
#define D_MODEL      64
#define N_HEADS       8
#define HEAD_DIM    (D_MODEL / N_HEADS) // must be even for RoPE
#define N_EXPERTS     4
#define TOP_K         2
#define HIDDEN_DIM   128   // per‑expert hidden size for SwiGLU

#if (D_MODEL % N_HEADS) != 0
#error "D_MODEL must be divisible by N_HEADS"
#endif
#if (HEAD_DIM % 2) != 0
#error "HEAD_DIM must be even for RoPE"
#endif

// ======= Utilities =======
static float frandf() { return (float)rand() / (float)RAND_MAX; }
static float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }

static void* xmalloc(size_t n) { void *p = malloc(n); if (!p) { fprintf(stderr, "OOM %zu\n", n); exit(1);} return p; }

// ======= Basic tensor helpers =======
typedef struct { int rows, cols; float *w; } Mat; // row‑major: rows x cols

static Mat mat_alloc(int rows, int cols) {
    Mat m; m.rows = rows; m.cols = cols; m.w = (float*)xmalloc(sizeof(float)*rows*cols); return m;
}
static void mat_free(Mat *m) { free(m->w); m->w = NULL; m->rows = m->cols = 0; }
static void mat_fill_rand(Mat *m, float scale) {
    for (int i=0;i<m->rows*m->cols;i++) m->w[i] = (frandf()*2.f - 1.f)*scale;
}
static inline float* mat_row(Mat *m, int r) { return &m->w[(size_t)r * m->cols]; }

// y = W x   where W:[o x i], x:[i]
static void matvec(float *restrict y, const Mat *restrict W, const float *restrict x) {
    int o = W->rows, i = W->cols; 
    for (int r=0;r<o;r++) {
        const float *w = &W->w[(size_t)r*i];
        float acc = 0.f;
        for (int c=0;c<i;c++) acc += w[c] * x[c];
        y[r] = acc;
    }
}

// y += x
static void vec_add_inplace(float *restrict y, const float *restrict x, int n) {
    for (int i=0;i<n;i++) y[i] += x[i];
}
// y = x
static void vec_copy(float *restrict y, const float *restrict x, int n) {
    memcpy(y, x, n*sizeof(float));
}
// y = 0
static void vec_zero(float *y, int n) { memset(y, 0, n*sizeof(float)); }

static float dot(const float *a, const float *b, int n) {
    float s=0.f; for (int i=0;i<n;i++) s += a[i]*b[i]; return s; }

// stable softmax in place
static void softmax_ip(float *x, int n) {
    float m = x[0];
    for (int i=1;i<n;i++) if (x[i] > m) m = x[i];
    float s=0.f; for (int i=0;i<n;i++) { x[i] = expf(x[i]-m); s += x[i]; }
    float inv = 1.f / s; for (int i=0;i<n;i++) x[i] *= inv;
}

// RMSNorm: y = x * gamma / sqrt(mean(x^2) + eps)
static void rmsnorm(float *restrict y, const float *restrict x, const float *restrict gamma, int n, float eps) {
    float ms = 0.f; for (int i=0;i<n;i++) ms += x[i]*x[i];
    ms /= (float)n; float inv = 1.f / sqrtf(ms + eps);
    for (int i=0;i<n;i++) y[i] = x[i] * inv * gamma[i];
}

static inline float silu(float x) { return x / (1.f + expf(-x)); }

// ======= Rotary Positional Embeddings (RoPE) for q/k per head =======
// theta base = 10000.0f like LLaMA‑style. Applies in place.
static void rope_apply(float *q_head, float *k_head, int head_dim, int pos) {
    // rotate dimension pairs (even, odd)
    for (int i=0;i<head_dim; i+=2) {
        int j = i/2; // pair index
        float inv_freq = powf(10000.0f, -(2.0f * j) / (float)head_dim);
        float ang = (float)pos * inv_freq;
        float cs = cosf(ang), sn = sinf(ang);
        float q0 = q_head[i], q1 = q_head[i+1];
        float k0 = k_head[i], k1 = k_head[i+1];
        // [x_even, x_odd] -> rotation
        q_head[i]   =  q0*cs - q1*sn;
        q_head[i+1] =  q0*sn + q1*cs;
        k_head[i]   =  k0*cs - k1*sn;
        k_head[i+1] =  k0*sn + k1*cs;
    }
}

// ======= Model definition (1 layer for clarity) =======
typedef struct {
    // embeddings & head
    Mat tok_emb;          // [VOCAB x D_MODEL]
    Mat lm_head;          // [VOCAB x D_MODEL]

    // attention projections
    Mat Wq, Wk, Wv, Wo;   // all [D_MODEL x D_MODEL]

    // per‑sublayer RMSNorm scale
    float *attn_norm;     // [D_MODEL]
    float *ffn_norm;      // [D_MODEL]

    // router
    Mat W_r;              // [N_EXPERTS x D_MODEL]
    float *b_r;           // [N_EXPERTS]

    // experts (SwiGLU)
    Mat W_up[N_EXPERTS];   // [HIDDEN_DIM x D_MODEL]
    Mat W_gate[N_EXPERTS]; // [HIDDEN_DIM x D_MODEL]
    Mat W_down[N_EXPERTS]; // [D_MODEL x HIDDEN_DIM]
} Model;

static void model_alloc(Model *M) {
    M->tok_emb = mat_alloc(VOCAB, D_MODEL);
    M->lm_head = mat_alloc(VOCAB, D_MODEL);
    M->Wq = mat_alloc(D_MODEL, D_MODEL);
    M->Wk = mat_alloc(D_MODEL, D_MODEL);
    M->Wv = mat_alloc(D_MODEL, D_MODEL);
    M->Wo = mat_alloc(D_MODEL, D_MODEL);

    M->attn_norm = (float*)xmalloc(sizeof(float)*D_MODEL);
    M->ffn_norm  = (float*)xmalloc(sizeof(float)*D_MODEL);

    M->W_r = mat_alloc(N_EXPERTS, D_MODEL);
    M->b_r = (float*)xmalloc(sizeof(float)*N_EXPERTS);

    for (int e=0;e<N_EXPERTS;e++) {
        M->W_up[e]   = mat_alloc(HIDDEN_DIM, D_MODEL);
        M->W_gate[e] = mat_alloc(HIDDEN_DIM, D_MODEL);
        M->W_down[e] = mat_alloc(D_MODEL, HIDDEN_DIM);
    }
}

static void model_free(Model *M) {
    mat_free(&M->tok_emb); mat_free(&M->lm_head);
    mat_free(&M->Wq); mat_free(&M->Wk); mat_free(&M->Wv); mat_free(&M->Wo);
    free(M->attn_norm); free(M->ffn_norm);
    mat_free(&M->W_r); free(M->b_r);
    for (int e=0;e<N_EXPERTS;e++) { mat_free(&M->W_up[e]); mat_free(&M->W_gate[e]); mat_free(&M->W_down[e]); }
}

static void model_init_random(Model *M, unsigned seed) {
    srand(seed);
    mat_fill_rand(&M->tok_emb, 0.02f);
    mat_fill_rand(&M->lm_head, 0.02f);
    mat_fill_rand(&M->Wq, 0.02f);
    mat_fill_rand(&M->Wk, 0.02f);
    mat_fill_rand(&M->Wv, 0.02f);
    mat_fill_rand(&M->Wo, 0.02f);

    for (int i=0;i<D_MODEL;i++) { M->attn_norm[i] = 1.0f; M->ffn_norm[i] = 1.0f; }

    mat_fill_rand(&M->W_r, 0.02f);
    for (int i=0;i<N_EXPERTS;i++) M->b_r[i] = 0.0f;

    for (int e=0;e<N_EXPERTS;e++) {
        mat_fill_rand(&M->W_up[e],   0.02f);
        mat_fill_rand(&M->W_gate[e], 0.02f);
        mat_fill_rand(&M->W_down[e], 0.02f);
    }
}

// ======= KV cache for attention =======
static float K_cache[MAX_SEQ][N_HEADS][HEAD_DIM];
static float V_cache[MAX_SEQ][N_HEADS][HEAD_DIM];

// ======= Attention forward (single step at pos) =======
static void attn_forward(Model *M, const float *x_in, int pos, float *x_out) {
    // Pre‑norm
    float nx[D_MODEL]; rmsnorm(nx, x_in, M->attn_norm, D_MODEL, 1e-5f);

    // Project to q, k, v
    float q[D_MODEL], k[D_MODEL], v[D_MODEL];
    matvec(q, &M->Wq, nx); matvec(k, &M->Wk, nx); matvec(v, &M->Wv, nx);

    // Split into heads and apply RoPE; write k/v into cache for this position
    for (int h=0; h<N_HEADS; h++) {
        float *qh = &q[h*HEAD_DIM];
        float *kh = &k[h*HEAD_DIM];
        float *vh = &v[h*HEAD_DIM];
        rope_apply(qh, kh, HEAD_DIM, pos);
        // store k,v for this step
        for (int d=0; d<HEAD_DIM; d++) {
            K_cache[pos][h][d] = kh[d];
            V_cache[pos][h][d] = vh[d];
        }
    }

    // Scaled dot‑product attention using cached K,V over positions [0..pos]
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    float out_heads[D_MODEL]; vec_zero(out_heads, D_MODEL);

    for (int h=0; h<N_HEADS; h++) {
        const float *qh = &q[h*HEAD_DIM];
        float attn[MAX_SEQ]; // scores 
        for (int t=0; t<=pos; t++) attn[t] = scale * dot(qh, K_cache[t][h], HEAD_DIM);
        softmax_ip(attn, pos+1);
        float oh[HEAD_DIM] = {0};
        for (int t=0; t<=pos; t++) {
            float a = attn[t];
            for (int d=0; d<HEAD_DIM; d++) oh[d] += a * V_cache[t][h][d];
        }
        // write into out_heads
        memcpy(&out_heads[h*HEAD_DIM], oh, sizeof(float)*HEAD_DIM);
    }

    // Output projection and residual add
    float proj[D_MODEL]; matvec(proj, &M->Wo, out_heads);
    vec_copy(x_out, x_in, D_MODEL); // residual
    vec_add_inplace(x_out, proj, D_MODEL);
}

// ======= MoE forward (top‑2, SwiGLU) =======
static void moe_forward(Model *M, const float *x_in, float *x_out) {
    // Pre‑norm
    float nx[D_MODEL]; rmsnorm(nx, x_in, M->ffn_norm, D_MODEL, 1e-5f);

    // Router logits
    float rlogits[N_EXPERTS]; matvec(rlogits, &M->W_r, nx); for (int e=0;e<N_EXPERTS;e++) rlogits[e] += M->b_r[e];
    softmax_ip(rlogits, N_EXPERTS);

    // Pick top‑2
    int i1=0, i2=1; if (rlogits[i2] > rlogits[i1]) { int tmp=i1; i1=i2; i2=tmp; }
    for (int i=2;i<N_EXPERTS;i++) {
        if (rlogits[i] > rlogits[i1]) { i2 = i1; i1 = i; }
        else if (rlogits[i] > rlogits[i2]) { i2 = i; }
    }
    float w1 = rlogits[i1], w2 = rlogits[i2];
    float wn = w1 + w2; if (wn <= 1e-9f) { w1 = 0.5f; w2 = 0.5f; } else { w1/=wn; w2/=wn; }

    float y1[D_MODEL], y2[D_MODEL];
    // expert e output: down( silu(up(nx)) ⊙ gate(nx) )
    for (int pass=0; pass<2; pass++) {
        int e = (pass==0 ? i1 : i2);
        float up[HIDDEN_DIM], gt[HIDDEN_DIM], mix[HIDDEN_DIM];
        matvec(up, &M->W_up[e], nx);
        matvec(gt, &M->W_gate[e], nx);
        for (int i=0;i<HIDDEN_DIM;i++) mix[i] = silu(up[i]) * gt[i];
        (pass==0 ? (void)matvec(y1, &M->W_down[e], mix) : (void)matvec(y2, &M->W_down[e], mix));
    }

    float y[D_MODEL]; for (int i=0;i<D_MODEL;i++) y[i] = w1*y1[i] + w2*y2[i];

    // Residual add
    vec_copy(x_out, x_in, D_MODEL);
    vec_add_inplace(x_out, y, D_MODEL);
}

// ======= One full layer step (attn + MoE) =======
static void layer_step(Model *M, int pos, int token_id, float *state_x, float *logits_out) {
    // Get token embedding
    const float *emb = mat_row(&M->tok_emb, token_id);
    if (pos == 0) vec_copy(state_x, emb, D_MODEL); else vec_add_inplace(state_x, emb, D_MODEL); // simple residual accumulation

    // Attention
    float x_after_attn[D_MODEL];
    attn_forward(M, state_x, pos, x_after_attn);

    // MoE
    float x_after_moe[D_MODEL];
    moe_forward(M, x_after_attn, x_after_moe);

    // LM head: logits = W_out x
    matvec(logits_out, &M->lm_head, x_after_moe);

    // Update state to post‑MoE for next token
    vec_copy(state_x, x_after_moe, D_MODEL);
}

// ======= Greedy decode demo (with random weights it’s nonsense) =======
static int argmax(const float *x, int n) { int m=0; for (int i=1;i<n;i++) if (x[i]>x[m]) m=i; return m; }

int main(int argc, char **argv) {
    Model M; model_alloc(&M); model_init_random(&M, 42);

    // zero KV cache
    for (int t=0;t<MAX_SEQ;t++) for (int h=0;h<N_HEADS;h++) for (int d=0;d<HEAD_DIM;d++) { K_cache[t][h][d]=0; V_cache[t][h][d]=0; }

    // parse input token ids from argv[1]
    int seq[MAX_SEQ]; int T=0;
    if (argc >= 2) {
        char *s = argv[1];
        while (*s) {
            while (*s==' '||*s=='\t') s++;
            if (!*s) break; 
            int id = atoi(s);
            if (id < 0 || id >= VOCAB) { fprintf(stderr, "token id %d out of range [0,%d)\n", id, VOCAB); return 1; }
            if (T >= MAX_SEQ) { fprintf(stderr, "too many tokens (>%d)\n", MAX_SEQ); return 1; }
            seq[T++] = id;
            while (*s && *s!=' ' && *s!='\t') s++;
        }
    } else {
        // default toy prompt
        int tmp[] = {1,5,4,2,3}; memcpy(seq, tmp, sizeof(tmp)); T = sizeof(tmp)/sizeof(tmp[0]);
    }

    float x[D_MODEL]; vec_zero(x, D_MODEL);
    float logits[VOCAB];

    for (int t=0; t<T; t++) {
        int tok = seq[t];
        if (t >= MAX_SEQ) break;
        layer_step(&M, t, tok, x, logits);
    }

    // Print top‑5 logits indices
    int idx[5] = {0,1,2,3,4};
    // simple partial sort
    for (int i=0;i<5;i++) idx[i] = i;
    for (int i=5;i<VOCAB;i++) {
        int m = 0; for (int j=1;j<5;j++) if (logits[idx[j]] < logits[idx[m]]) m=j;
        if (logits[i] > logits[idx[m]]) idx[m] = i;
    }
    // order descending
    for (int i=0;i<5;i++) for (int j=i+1;j<5;j++) if (logits[idx[j]] > logits[idx[i]]) { int tmp=idx[i]; idx[i]=idx[j]; idx[j]=tmp; }

    printf("Top‑5 next token logits (id:logit):\n");
    for (int i=0;i<5;i++) printf("  %d : %.4f\n", idx[i], logits[idx[i]]);

    model_free(&M);
    return 0;
}
