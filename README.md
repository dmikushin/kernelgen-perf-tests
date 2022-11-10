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

## Example output

```
> ./benchmark
Usage: benchmark <nx> <ny> <ns> <niters> <nruns> <target> [<target>]
Example: kernelgen_runmode=1 kernelgen_szheap=$((1024*1024*800)) kernelgen_verbose=$((1<<6)) ./benchmark 512 256 256 10 10 kernelgen gcc
                                      ^
> bash -c 'kernelgen_runmode=1 kernelgen_szheap=$((1024*1024*800)) kernelgen_verbose=$((1<<6)) ./benchmark 512 256 256 10 10 cuda'
Found test divergence
Found test gameoflife
Found test gaussblur
Found test gradient
Found test jacobi
Found test lapgsrb
Found test laplacian
Found test matmul
Found test matvec
Found test sincos
Found test tricubic
Found test tricubic2
Found test uxx1
Found test vecadd
Found test wave13pt
Found test whispering
---- Using the following CUDA driver: ----
NVRM version: NVIDIA UNIX x86_64 Kernel Module  520.61.05  Thu Sep 29 05:30:25 UTC 2022
GCC version:  gcc version 11.3.0 (Ubuntu 11.3.0-1ubuntu1~22.04) 
------------------------------------------

---- Using the following GPU(s): ----
GPU 0: NVIDIA T500 (UUID: GPU-cc2d492e-bb5f-3ee8-23a2-f45df43db985)
-------------------------------------

---- Using the following NVIDIA compiler: ----
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
-------------------------------------------

--------------
| RUN #0     |
----------------------------------------------------------------------------------------------------------------------------------------------------
|       test |                target |   i_mean |   t_init |  t_alloc |   t_load |   t_comp |    t_krn | nreg_krn |   t_save |   t_free |   f_mean |
----------------------------------------------------------------------------------------------------------------------------------------------------
| divergence |                  cuda | 2.2e-05  | 0.051457 | 0.122236 | 0.349886 | 0.200252 | 0.019214 | 37       | 0.082368 | 0.000834 | -3.2e-05 |
| gameoflife |                  cuda | 4.1e-05  | 0.02813  | 0.071689 | 0.176462 | 0.1308   | 0.013059 | 32       | 0.082304 | 0.000442 | 1.2e-05  |
|  gaussblur |                  cuda | 4.1e-05  | 0.029973 | 0.071299 | 0.176605 | 0.278899 | 0.027867 | 64       | 0.082493 | 0.000892 | 0.000101 |
|   gradient |                  cuda | 2.2e-05  | 0.029496 | 0.076196 | 0.349551 | 0.245132 | 0.024484 | 36       | 0.241224 | 0.000772 | -5e-06   |
|     jacobi |                  cuda |     FAIL |      N/A |      N/A |      N/A |     FAIL |      N/A |      N/A |      N/A |      N/A |     FAIL |
|    lapgsrb |                  cuda | 4.1e-05  | 0.027912 | 0.071739 | 0.174276 | 0.257997 | 0.025772 | 64       | 0.082375 | 0.000465 | 0.009615 |
|  laplacian |                  cuda | 4.1e-05  | 0.027572 | 0.060741 | 0.173186 | 0.153463 | 0.015318 | 35       | 0.082499 | 0.000463 | 1.1e-05  |
|     matmul |                  cuda |     FAIL |      N/A |      N/A |      N/A |     FAIL |      N/A |      N/A |      N/A |      N/A |     FAIL |
|     matvec |                  cuda | 0.022473 | 0.027863 | 0.072959 | 0.086674 | 0.098686 | 0.009848 | 27       | 0.000219 | 0.000309 | -0.03619 |
|     sincos |                  cuda |     FAIL |      N/A |      N/A |      N/A |     FAIL |      N/A |      N/A |      N/A |      N/A |     FAIL |
|   tricubic |                  cuda | -8e-06   | 0.02941  | 0.099977 | 0.438189 | 1.38882  | 0.138859 | 96       | 0.082389 | 0.000954 | 0.000443 |
|  tricubic2 |                  cuda | -8e-06   | 0.030427 | 0.082373 | 0.437034 |     FAIL |      N/A |      N/A |      N/A |      N/A |     FAIL |
|       uxx1 |                  cuda | -2.3e-05 | 0.045117 | 0.076821 | 0.523423 | 0.40396  | 0.040363 | 52       | 0.082218 | 0.001094 | -0.03037 |
|     vecadd |                  cuda | 2.4e-05  | 0.029357 | 0.071748 | 0.262517 | 0.117928 | 0.011773 | 17       | 0.082429 | 0.000671 | -0.0037  |
|   wave13pt |                  cuda | 2.4e-05  | 0.027651 | 0.074739 | 0.26198  | 0.270446 | 0.027033 | 56       | 0.08246  | 0.000689 | 0.000173 |
| whispering |                  cuda |     FAIL |      N/A |      N/A |      N/A |     FAIL |      N/A |      N/A |      N/A |      N/A |     FAIL |
----------------------------------------------------------------------------------------------------------------------------------------------------
```

