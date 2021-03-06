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
szreal ?=
ifeq ($(real), double)
szreal ?= -fdefault-real-8
endif

# GCC compiler/linker options
GCC ?= gcc -O3 -D_GNU_SOURCE -std=c99
GFORTRAN ?= gfortran -O3 $(szreal)

# Default problem domain dimensions
NX ?= 512
NY ?= 256
NS ?= 256

all: jacobi

jacobi: main.o jacobi.o timing.o
	$(SILENT)$(GFORTRAN) $^ -o $@ -lrt

main.o: ../main.c ../timing.h
	$(SILENT)$(GCC) -Dreal=$(real) -c $< -o $@

jacobi.o: ../jacobi.F90 ../timing.h
	$(SILENT)$(GFORTRAN) -c $< -o $@

timing.o: ../timing.c ../timing.h
	$(SILENT)$(GCC) -c $< -o $@

clean:
	$(SILENT)rm -rf *.o jacobi

test: jacobi
	$(SILENT)./$< $(NX) $$(($(NY) * $(NS))) $(NT)

