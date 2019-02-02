# KernelGen Stencil Performance Test Suite for CPU and GPU compilers

This test suite is developed as a part of KernelGen project:

```bibtex
@inproceedings{Mikushin:2014:KDI:2672598.2672916,
 author = {Mikushin, Dmitry and Likhogrud, Nikolay and Zhang, Eddy Z. and Bergstr\"{o}m, Christopher},
 title = {KernelGen  --  The Design and Implementation of a Next Generation Compiler Platform for Accelerating Numerical Models on GPUs},
 booktitle = {Proceedings of the 2014 IEEE International Parallel \& Distributed Processing Symposium Workshops},
 series = {IPDPSW '14},
 year = {2014},
 isbn = {978-1-4799-4116-2},
 pages = {1011--1020},
 numpages = {10},
 url = {http://dx.doi.org/10.1109/IPDPSW.2014.115},
 doi = {10.1109/IPDPSW.2014.115},
 acmid = {2672916},
 publisher = {IEEE Computer Society},
 address = {Washington, DC, USA},
 keywords = {GPU, LLVM, OpenACC, JIT-compilation, stencils},
}
```

## Overview

The test suite targets host CPUs, NVIDIA GPUs and Intel Xeon Phi accelerators with the following compiler toolchains:

 * `caps` : CAPS Enterprise Compiler, OpenACC backend (defunct)
 * `cuda` : NVIDIA CUDA Compiler
 * `gcc` : GCC compiler for host CPU
 * `kernelgen` : KernelGen LLVM-based parallelizing compiler
 * `mic` : Intel Compiler for Xeon Phi accelerator board
 * `pathscale` : PathScale ENZO, OpenACC backend (defunct)
 * `patus` : PATUS Framework for Parallel Iterative Stencil Computations
 * `pgi` : PGI OpenACC Compiler (now owned by NVIDIA)

## Prerequisites

Make sure the toolchains intended for performance testsing are available from the command line, e.g. GCC compiler, NVIDIA `nvcc` compiler, PGI `pgcc` and `pgfortran` compilers and so on.

If you plan to plot performance results, install `gnuplot`:

```
sudo apt install gnuplot
```

## Deployment

Build every test of the suite for selected toolchains with e.g. the following `make` command:

```
make gcc cuda
```

Run tests for selected toolchains and dump performance results into a file:

```
make benchmark TARGETS="cuda gcc" >TEST.double.gtx1060m
```

Plot performance results into a pdf figure:

```
./mkchart -second2all -ylabel "Absolute speedup against CPU version" TEST.double.gtx1060m -o TEST.double.gtx1060m.pdf
```
