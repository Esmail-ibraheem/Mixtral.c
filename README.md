# Mixtral.c
<img width="1024" height="1024" alt="ChatGPT Image Aug 19, 2025, 06_14_58 PM" src="https://github.com/user-attachments/assets/425b96f9-a466-4fa0-bee8-d26e2d6989fe" />

mixtral model from scratch using pure C

```
mixtral/
├── include/                # Public headers
│   ├── tensor.h
│   ├── allocator.h
│   ├── model.h
│   ├── attention.h
│   ├── layernorm.h
│   ├── activations.h
│   ├── tokenizer.h
│   ├── sampler.h
│   └── io.h
├── src/                    # Implementation files
│   ├── allocator.c
│   ├── tensor.c
│   ├── blas_kernels.c
│   ├── attention.c
│   ├── layernorm.c
│   ├── activations.c
│   ├── tokenizer.c
│   ├── sampler.c
│   ├── io.c
│   └── main_inference.c
├── tools/                  # Build scripts, weight converters
│   ├── convert_weights.c
│   └── build.sh
├── tests/                  # Unit & integration tests
│   ├── test_tensor.c
│   ├── test_attention.c
│   └── …
├── benchmarks/             # Perf harnesses
│   └── perf_gemm.c
├── third_party/            # Optional: BLAS, JSON, ICU
└── Makefile

```


## How to build:
```
  gcc -O3 -march=native inference.c -o inference -lm
```

**How to run (toy demo with random weights):**
```
   ./inference "1 5 4 2 3"   # space‑separated token ids from 0..VOCAB-1
```
