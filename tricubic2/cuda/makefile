##===----------------------------------------------------------------------===##
##
##     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
##        compiler for NVIDIA GPUs, targeting numerical modeling code.
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##
##===----------------------------------------------------------------------===##

-include ../../makefile.in

# Verbose setting
VERBOSE ?= 0
SILENT ?= @
ifeq ($(VERBOSE), 1)
SILENT =
endif

# Floating-point type: float or double
real ?= float

# PGI/OpenACC compiler/linker options
GPUARCH ?= 20
GPUCUDA ?= 5.0
NVCC ?= nvcc -I. -I.. -O3 -arch=sm_$(GPUARCH)
PGFC ?= pgfortran -O3 $(pgi_szreal)

# Default problem domain dimensions
NX ?= 512
NY ?= 256
NS ?= 256

all: tricubic2

tricubic2: tricubic2.o timing.o cuda_profiling.o
	$(SILENT)$(NVCC) $^ -o $@ -lrt -Xlinker --wrap=cudaLaunch -Xlinker --wrap=cudaLaunchKernel -Xcompiler -rdynamic --cudart=shared -ldl

tricubic2.o: ../tricubic2.cu ../timing.h cuda_profiling.h
	$(SILENT)$(NVCC) -Dreal=$(real) -c $< -o $@

timing.o: ../timing.c ../timing.h
	$(SILENT)$(NVCC) -c $< -o $@

cuda_profiling.o: cuda_profiling.cu cuda_profiling.h
	$(SILENT)$(NVCC) -c $< -o $@

clean:
	$(SILENT)rm -rf *.o tricubic2

test: tricubic2
	$(SILENT)./$< $(NX) $(NY) $(NS) $(NT)

