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
pgi_szreal =
ifeq ($(real), double)
pgi_szreal = -r8
endif

# PGI/OpenACC compiler/linker options
GPUARCH ?= 20
GPUCUDA ?= 5.0
PGF90_CUDA ?= pgf90 -O3 -Mcuda=keepgpu,keepbin,keepptx,ptxinfo,cc$(GPUARCH),cuda$(GPUCUDA) $(pgi_szreal) -D__CUDAFOR__
NVCC ?= nvcc -I. -I.. -O3 -arch=sm_$(GPUARCH)

# NVIDIA compile options
NVCC ?= nvcc -O3

# Default problem domain dimensions
NX ?= 512
NY ?= 256
NS ?= 256

all: jacobi

jacobi: main.o jacobi.o timing.o cuda_profiling.o
	$(SILENT)$(PGF90_CUDA) -Mnomain $^ $(LIBWRAPPER) -o $@ -lrt -L$(shell dirname $(shell which nvcc))/../lib64 -lcudart -lcuda -lstdc++ -lgcc_s -Wl,--wrap=cudaLaunch -Wl,--export-dynamic -ldl

main.o: ../main.cu ../timing.h
	$(SILENT)$(NVCC) -Dreal=$(real) -c $< -o $@

jacobi.o: ../jacobi.CUF
	$(SILENT)$(PGF90_CUDA) -c $< -o $@

timing.o: ../timing.c ../timing.h
	$(SILENT)$(PGCC_CUDA) -c $< -o $@

cuda_profiling.o: cuda_profiling.cu cuda_profiling.h
	$(SILENT)$(NVCC) -c $< -o $@

clean:
	$(SILENT)rm -rf *.o jacobi *.ptx *.gpu *.bin

test: jacobi
	$(SILENT)./$< $(NX) $$(($(NY) * $(NS))) $(NT)

