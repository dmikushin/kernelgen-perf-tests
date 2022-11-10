//===----------------------------------------------------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

#include "timing.h"

#if defined(_OPENACC)
#include <openacc.h>
#include "openacc_profiling.h"
#endif

#if defined(__CUDACC__)
#include "cuda_profiling.h"
#endif

// Memory alignment, for vectorization on MIC.
// 4096 should be best for memory transfers over PCI-E.
#define MEMALIGN 4096

#define _A(array, ix, iy) (array[(ix) + nx * (iy)])

#if defined(_MIC)
__attribute__((target(mic)))
#endif
#if defined(__CUDACC__)
extern "C" __global__
#endif
void matvec(int nx, int ny,
#if defined(__CUDACC__)
	kernelgen_cuda_config_t config,
#endif
	real* A, real* x, real* y)
{
#if defined(__CUDACC__)
	#define i_stride (config.strideDim.x)
	#define i_offset (blockIdx.x * blockDim.x + threadIdx.x)
	#define i_increment i_stride
#else
	#define i_offset 0
	#define i_increment 1
#endif
#if defined(_OPENACC)
	size_t szarray = (size_t)nx * ny;
	#pragma acc kernels loop independent present(A[0:szarray], x[0:nx], y[0:ny])
#endif
#if defined(_OPENMP) || defined(_MIC)
	#pragma omp parallel for
#endif
	for (int j = 0 + i_offset; j < ny; j += i_increment)
	{
		y[j] = 0;
#if defined(_OPENACC)
		#pragma acc loop seq
#endif
		for (int i = 0; i < nx; i++)
		{
			y[j] += _A(A, i, j) * x[i];
		}
	}
}

#define parse_arg(name, arg) \
	int name = atoi(arg); \
	if (name < 0) \
	{ \
		printf("Value for " #name " is invalid: %d\n", name); \
		exit(1); \
	}

#define real_rand() (((real)(rand() / (double)RAND_MAX) - 0.5) * 2)

int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		printf("Usage: %s <nx> <ny> <nt>\n", argv[0]);
		exit(1);
	}

	const char* no_timing = getenv("NO_TIMING");

#if defined(_OPENACC) || defined(__CUDACC__)
	char* regcount_fname = getenv("PROFILING_FNAME");
	if (regcount_fname)
	{
		char* regcount_lineno = getenv("PROFILING_LINENO");
		int lineno = -1;
		if (regcount_lineno)
			lineno = atoi(regcount_lineno);
		kernelgen_enable_regcount(regcount_fname, lineno);
	}
#endif

	parse_arg(nx, argv[1]);
	parse_arg(ny, argv[2]);
	parse_arg(nt, argv[3]);

	real* A = (real*)memalign(MEMALIGN, nx * ny * sizeof(real));
	real* x = (real*)memalign(MEMALIGN, nx * sizeof(real));
	real* y = (real*)memalign(MEMALIGN, ny * sizeof(real));

	if (!A || !x || !y)
	{
		printf("Error allocating memory for arrays: %p, %p, %p\n", A, x, y);
		exit(1);
	}

	real amean = 0.0f, xmean = 0.0f, ymean = 0.0f;
	for (int i = 0; i < nx * ny; i++)
	{
		A[i] = real_rand();
		amean += A[i];
	}
	for (int i = 0; i < nx; i++)
	{
		x[i] = real_rand();
		xmean += x[i];
	}
	for (int i = 0; i < ny; i++)
	{
		y[i] = real_rand();
		ymean += y[i];
	}
	if (!no_timing) printf("initial mean = %f\n", amean / (nx * ny) + xmean / nx + ymean / ny);

	//
	// MIC or OPENACC or CUDA:
	//
	// 1) Perform an empty offload, that should strip
	// the initialization time from further offloads.
	//
#if defined(_MIC) || defined(_OPENACC) || defined(__CUDACC__)
	volatile struct timespec init_s, init_f;
#if defined(_MIC)
	get_time(&init_s);
	#pragma offload target(mic) \
		nocopy(A:length(nx * ny) alloc_if(0) free_if(0)), \
		nocopy(x:length(nx) alloc_if(0) free_if(0)), \
		nocopy(y:length(ny) alloc_if(0) free_if(0))
	{ }
	get_time(&init_f);
#endif
#if defined(_OPENACC)
	get_time(&init_s);
#if defined(__PGI)
	acc_init(acc_device_nvidia);
#else
	acc_init(acc_device_gpu);
#endif
	get_time(&init_f);
#endif
#if defined(__CUDACC__)
	get_time(&init_s);
	int count = 0;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&count));
	get_time(&init_f);
#endif
	double init_t = get_time_diff((struct timespec*)&init_s, (struct timespec*)&init_f);
	if (!no_timing) printf("init time = %f sec\n", init_t);
#endif

	//
	// MIC or OPENACC or CUDA:
	//
	// 2) Allocate data on device, but do not copy anything.
	//
#if defined(_MIC) || defined(_OPENACC) || defined(__CUDACC__)
	volatile struct timespec alloc_s, alloc_f;
#if defined(_MIC)
	get_time(&alloc_s);
	#pragma offload target(mic) \
		nocopy(A:length(nx * ny) alloc_if(1) free_if(0)), \
		nocopy(x:length(nx) alloc_if(1) free_if(0)), \
		nocopy(y:length(ny) alloc_if(1) free_if(0))
	{ }
	get_time(&alloc_f);
#endif
#if defined(_OPENACC)
	get_time(&alloc_s);
	#pragma acc data create (A[0:nx*ny], x[0:nx], y[0:ny])
	{
	get_time(&alloc_f);
#endif
#if defined(__CUDACC__)
	get_time(&alloc_s);
	real *A_dev = NULL, *x_dev = NULL, *y_dev = NULL;
	CUDA_SAFE_CALL(cudaMalloc(&A_dev, nx * ny * sizeof(A[0])));
	CUDA_SAFE_CALL(cudaMalloc(&x_dev, nx * sizeof(x[0])));
	CUDA_SAFE_CALL(cudaMalloc(&y_dev, ny * sizeof(y[0])));
	get_time(&alloc_f);
#endif
	double alloc_t = get_time_diff((struct timespec*)&alloc_s, (struct timespec*)&alloc_f);
	if (!no_timing) printf("device buffer alloc time = %f sec\n", alloc_t);
#endif

	//
	// MIC or OPENACC or CUDA:
	//
	// 3) Transfer data from host to device and leave it there,
	// i.e. do not allocate deivce memory buffers.
	//
#if defined(_MIC) || defined(_OPENACC) || defined(__CUDACC__)
	volatile struct timespec load_s, load_f;
#if defined(_MIC)
	get_time(&load_s);
	#pragma offload target(mic) \
		in(A:length(nx * ny) alloc_if(0) free_if(0)), \
		in(x:length(nx) alloc_if(0) free_if(0)), \
		in(y:length(ny) alloc_if(0) free_if(0))
	{ }
	get_time(&load_f);
#endif
#if defined(_OPENACC)
	get_time(&load_s);
	#pragma acc update device(A[0:nx*ny], x[0:nx], y[0:ny])
	get_time(&load_f);
#endif
#if defined(__CUDACC__)
	get_time(&load_s);
	CUDA_SAFE_CALL(cudaMemcpy(A_dev, A, nx * ny * sizeof(A[0]), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(x_dev, x, nx * sizeof(x[0]), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(y_dev, y, ny * sizeof(y[0]), cudaMemcpyHostToDevice));
	get_time(&load_f);
#endif
	double load_t = get_time_diff((struct timespec*)&load_s, (struct timespec*)&load_f);
	if (!no_timing) printf("data load time = %f sec (%f GB/sec)\n", load_t,
		((nx * ny + nx + ny) * sizeof(real)) / (load_t * 1024 * 1024 * 1024));
#endif

	//
	// 4) Perform data processing iterations, keeping all data
	// on device.
	//
	volatile struct timespec compute_s, compute_f;
	get_time(&compute_s);
#if defined(_MIC)
	#pragma offload target(mic) \
		nocopy(A:length(nx * ny) alloc_if(0) free_if(0)), \
		nocopy(x:length(nx) alloc_if(0) free_if(0)), \
		nocopy(y:length(ny) alloc_if(0) free_if(0))
#endif
#if defined(__CUDACC__)
	kernelgen_cuda_config_t config;
	kernelgen_cuda_configure_gird(1, nx, 1, 1, &config);
#endif
	{
#if !defined(__CUDACC__)
		real *Ap = A, *xp = x, *yp = y;
#else
		real *Ap = A_dev, *xp = x_dev, *yp = y_dev;
#endif
		for (int it = 0; it < nt; it++)
#if !defined(__CUDACC__)
			matvec(nx, ny, Ap, xp, yp);
#else
			matvec<<<config.gridDim, config.blockDim>>>(
				nx, ny,
				config,
				Ap, xp, yp);
			CUDA_SAFE_CALL(cudaGetLastError());
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
	}
	get_time(&compute_f);
	double compute_t = get_time_diff((struct timespec*)&compute_s, (struct timespec*)&compute_f);
	if (!no_timing) printf("compute time = %f sec\n", compute_t);

	//
	// MIC or OPENACC or CUDA:
	//
	// 5) Transfer output data back from device to host.
	//
#if defined(_MIC) || defined(_OPENACC) || defined(__CUDACC__)
	volatile struct timespec save_s, save_f;
#if defined(_MIC)
	get_time(&save_s);
	#pragma offload target(mic) \
		out(y:length(ny) alloc_if(0) free_if(0))
	{ }
	get_time(&save_f);
#endif
#if defined(_OPENACC)
	get_time(&save_s);
	#pragma acc update host (y[0:ny])
	get_time(&save_f);
#endif
#if defined(__CUDACC__)
	get_time(&save_s);
	CUDA_SAFE_CALL(cudaMemcpy(y, y_dev, ny * sizeof(y[0]), cudaMemcpyDeviceToHost));
	get_time(&save_f);
#endif
	double save_t = get_time_diff((struct timespec*)&save_s, (struct timespec*)&save_f);
	if (!no_timing) printf("data save time = %f sec (%f GB/sec)\n", save_t, (ny * sizeof(real)) / (save_t * 1024 * 1024 * 1024));
#endif

	//
	// MIC or OPENACC or CUDA:
	//
	// 6) Deallocate device data buffers.
	// OPENACC does not seem to have explicit deallocation.
	//
#if defined(_OPENACC)
	}
#endif
#if defined(_MIC) || defined(__CUDACC__)
	volatile struct timespec free_s, free_f;
#if defined(_MIC)
	get_time(&free_s);
	#pragma offload target(mic) \
		nocopy(A:length(nx * ny) alloc_if(0) free_if(1)), \
		nocopy(x:length(nx) alloc_if(0) free_if(1)), \
		nocopy(y:length(ny) alloc_if(0) free_if(1))
	{ }
	get_time(&free_f);
#endif
#if defined(__CUDACC__)
	get_time(&free_s);
	CUDA_SAFE_CALL(cudaFree(A_dev));
	CUDA_SAFE_CALL(cudaFree(x_dev));
	CUDA_SAFE_CALL(cudaFree(y_dev));
	get_time(&free_f);
#endif
	double free_t = get_time_diff((struct timespec*)&free_s, (struct timespec*)&free_f);
	if (!no_timing) printf("device buffer free time = %f sec\n", free_t);
#endif

	ymean = 0.0f;
	for (int i = 0; i < ny; i++)
		ymean += y[i];
	printf("final mean = %f\n", ymean / ny);

	free(A);
	free(x);
	free(y);

	fflush(stdout);

	return 0;
}

