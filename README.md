# Mixtral.c
mixtral model from scratch using pure c

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
  gcc -O3 -march=native mixtral.c -o mixtral -lm
```

**How to run (toy demo with random weights):**
```
   ./mixtral "1 5 4 2 3"   # space‑separated token ids from 0..VOCAB-1
```
