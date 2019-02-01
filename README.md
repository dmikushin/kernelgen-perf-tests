# KernelGen Stencil Performance test suite

## Overview

The test suite targets host CPU, NVIDIA GPU and Intel Xeon Phi with the following compiler toolchains:

 * `caps` : CAPS Enterprise Compiler, OpenACC backend (defunct)
 * `cuda` : NVIDIA CUDA Compiler
 * `gcc` : GCC compiler for host CPU
 * `kernelgen` : KernelGen LLVM-based parallelizing compiler
 * `mic` : Intel Compiler for Xeon Phi accelerator board
 * `pathscale` : PathScale ENZO, OpenACC backend (defunct)
 * `patus` : PATUS Framework for Parallel Iterative Stencil Computations
 * `pgi` : PGI OpenACC Compiler (now owned by NVIDIA)

## Deployment

Build every test of the suite for selected toolchains with e.g. the following `make` command:

```
make gcc cuda
```

