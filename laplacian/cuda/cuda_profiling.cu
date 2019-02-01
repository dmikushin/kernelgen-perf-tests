//===----------------------------------------------------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>
#include <stdio.h>

using namespace std;

#include "cuda_profiling.h"
#include "timing.h"

static char* wrapper_funcname = 0;
static long wrapper_lineno = 0;

int kernelgen_enable_regcount(char* funcname, long lineno)
{
	wrapper_funcname = funcname;
	wrapper_lineno = lineno;
	return 0;
}

int kernelgen_disable_regcount()
{
	wrapper_funcname = 0;
	return 0;
}

int kernelgen_cuda_configure_gird(int nx, int ny, int ns,
	dim3* gridDim, dim3* blockDim, dim3* strideDim)
{
	// Block dimensions are fixed to the best values we've
	// found earlier during exhaustive search within KernelGen.
	blockDim->x = 128;
	blockDim->y = 1;
	blockDim->z = 1;

	strideDim->x = nx;
	strideDim->y = ny;
	strideDim->z = ns;
	
	// If maximum permitted GPU grid dimensions are large
	// enough to hold the original problem dimensions,
	// we just divide the problem size by block sizes.
	// Otherwise, the only solution is to assign multiple
	// grid points to each block thread. In order to comply
	// with coalescing requirements, we make a gap (stride)
	// between grid points indexes assigned to the same thread.
	gridDim->x = nx / blockDim->x;
	gridDim->y = ny / blockDim->y;
	gridDim->z = ns / blockDim->z;
	if (nx % blockDim->x) gridDim->x++;
	if (ny % blockDim->y) gridDim->y++;
	if (ns % blockDim->z) gridDim->z++;
	struct cudaDeviceProp props;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, 0));
	if (props.maxGridSize[0] * blockDim->x < nx)
	{
		gridDim->x = props.maxGridSize[0];
		strideDim->x = props.maxGridSize[0] * blockDim->x;
	}
	if (props.maxGridSize[1] * blockDim->y < ny)
	{
		gridDim->y = props.maxGridSize[1];
		strideDim->y = props.maxGridSize[1] * blockDim->y;
	}
	if (props.maxGridSize[2] * blockDim->z < ns)
	{
		gridDim->z = props.maxGridSize[2];
		strideDim->z = props.maxGridSize[2] * blockDim->z;
	}
	
	return 0;
}

extern "C" cudaError_t __real_cudaLaunch(const void *func);

extern "C" cudaError_t __wrap_cudaLaunch(const void *func)
{
	if (!wrapper_funcname)
		return __real_cudaLaunch(func);

	// Find out the kernel name.
	Dl_info info;
	if (dladdr(func, &info) == 0)
	{
		fprintf(stderr, "Error in dladdr(%p, %p): %s\n", func, &info, dlerror());
		exit(-1);
	}
	if (info.dli_saddr != func)
	{
		fprintf(stderr, "Cannot find kernel name for address %p\n", func);
		exit(-1);
	}
	const char* kernel_name = info.dli_sname;

	// Get the kernel register count.
	struct cudaFuncAttributes attrs;
	CUDA_SAFE_CALL(cudaFuncGetAttributes(&attrs, func));
	printf("%s regcount = %d\n", kernel_name, attrs.numRegs);

	// Get the kernel execution time.
	struct timespec start, finish;
	get_time(&start);
	cudaError_t result = __real_cudaLaunch(func);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	get_time(&finish);
	double kernel_time = get_time_diff(&start, &finish);
	if (kernel_name)
		printf("%s kernel time = %f\n", kernel_name, kernel_time);
	else
		printf("kernel time = %f\n", kernel_time);
	return result;
}

