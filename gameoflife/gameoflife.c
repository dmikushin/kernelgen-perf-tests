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

// PATUS-generated kernel declaration.
#if defined(_PATUS)
void gameoflife_patus(real** dummy, real* u0, real* u1, int nx, int ny);
#endif

#if defined(_MIC)
__attribute__((target(mic)))
#endif
#if defined(__CUDACC__)
extern "C" __global__
#endif
void gameoflife(int nx, int ny,
#if defined(__CUDACC__)
	int i_stride, int j_stride,
#endif
	real* u0, real* u1)
{
#if defined(__CUDACC__)
	#define j_offset (blockIdx.y * blockDim.y + threadIdx.y)
	#define i_offset (blockIdx.x * blockDim.x + threadIdx.x)
	#define j_increment j_stride
	#define i_increment i_stride
#else
	#define j_offset 0
	#define i_offset 0
	#define j_increment 1
	#define i_increment 1
#endif
#if defined(_PATUS)
	real* dummy;
	#pragma omp parallel
	gameoflife_patus(&dummy, u0, u1, nx, ny);
#else
#if defined(_OPENACC)
	size_t szarray = (size_t)nx * ny;
	#pragma acc kernels loop gang(65535), independent present(u0[0:szarray], u1[0:szarray])
#endif
#if defined(_OPENMP) || defined(_MIC)
	#pragma omp parallel for
#endif
	for (int j = 1 + j_offset; j < ny - 1; j += j_increment)
	{
#if defined(_OPENACC)
		#pragma acc loop independent vector(128)
#endif
		for (int i = 1 + i_offset; i < nx - 1; i += i_increment)
		{
			// Some large number
			real C = 100000000000000000000.;

			// Count the number of live neighbors
			real L =
				_A(u0, i-1, j-1) + _A(u0, i, j-1) + _A(u0, i+1, j-1) +
				_A(u0, i-1, j  )                  + _A(u0, i+1, j  ) +
				_A(u0, i-1, j+1) + _A(u0, i, j+1) + _A(u0, i+1, j+1); 

			// Apply the rules
			_A(u1, i, j) = 1. / (1. + (_A(u0, i, j) + L - 3.) * (L - 3.) * C);
		}
	}
#endif
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

	size_t szarray = (size_t)nx * ny;
	size_t szarrayb = szarray * sizeof(real);

	real* u0 = (real*)memalign(MEMALIGN, szarrayb);
	real* u1 = (real*)memalign(MEMALIGN, szarrayb);

	if (!u0 || !u1)
	{
		printf("Error allocating memory for arrays: %p, %p\n", u0, u1);
		exit(1);
	}

	real mean = 0.0f;
	for (int i = 0; i < szarray; i++)
	{
		u0[i] = real_rand();
		u1[i] = real_rand();
		mean += u0[i] + u1[i];
	}
	printf("initial mean = %f\n", mean / szarray / 2);

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
		nocopy(u0:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(u1:length(szarray) alloc_if(0) free_if(0))
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
		nocopy(u0:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(u1:length(szarray) alloc_if(1) free_if(0))
	{ }
	get_time(&alloc_f);
#endif
#if defined(_OPENACC)
	get_time(&alloc_s);
	#pragma acc data create (u0[0:szarray], u1[0:szarray])
	{
	get_time(&alloc_f);
#endif
#if defined(__CUDACC__)
	get_time(&alloc_s);
	real *u0_dev = NULL, *u1_dev = NULL;
	CUDA_SAFE_CALL(cudaMalloc(&u0_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&u1_dev, szarrayb));
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
		in(u0:length(szarray) alloc_if(0) free_if(0)), \
		in(u1:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&load_f);
#endif
#if defined(_OPENACC)
	get_time(&load_s);
	#pragma acc update device(u0[0:szarray], u1[0:szarray])
	get_time(&load_f);
#endif
#if defined(__CUDACC__)
	get_time(&load_s);
	CUDA_SAFE_CALL(cudaMemcpy(u0_dev, u0, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(u1_dev, u1, szarrayb, cudaMemcpyHostToDevice));
	get_time(&load_f);
#endif
	double load_t = get_time_diff((struct timespec*)&load_s, (struct timespec*)&load_f);
	if (!no_timing) printf("data load time = %f sec (%f GB/sec)\n", load_t, 2 * szarrayb / (load_t * 1024 * 1024 * 1024));
#endif

	//
	// 4) Perform data processing iterations, keeping all data
	// on device.
	//
	int idxs[] = { 0, 1 };
	volatile struct timespec compute_s, compute_f;
	get_time(&compute_s);
#if defined(_MIC)
	#pragma offload target(mic) \
		nocopy(u0:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(u1:length(szarray) alloc_if(0) free_if(0))
#endif
#if defined(__CUDACC__)
	dim3 gridDim, blockDim, strideDim;
	kernelgen_cuda_configure_gird(nx, ny, 1, &gridDim, &blockDim, &strideDim);
#endif
	{
#if !defined(__CUDACC__)
		real *u0p = u0, *u1p = u1;
#else
		real *u0p = u0_dev, *u1p = u1_dev;
#endif

		for (int it = 0; it < nt; it++)
		{
#if !defined(__CUDACC__)
			gameoflife(nx, ny, u0p, u1p);
#else
			gameoflife<<<gridDim, blockDim>>>(nx, ny,
				strideDim.x, strideDim.y,
				u0p, u1p);
			CUDA_SAFE_CALL(cudaGetLastError());
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
			real* w = u0p; u0p = u1p; u1p = w;
			int idx = idxs[0]; idxs[0] = idxs[1]; idxs[1] = idx;
		}
	}
	get_time(&compute_f);
	double compute_t = get_time_diff((struct timespec*)&compute_s, (struct timespec*)&compute_f);
	if (!no_timing) printf("compute time = %f sec\n", compute_t);

	real* u[] = { u0, u1 }; 
	u0 = u[idxs[0]]; u1 = u[idxs[1]];
#if defined(__CUDACC__)
	real* u_dev[] = { u0_dev, u1_dev };
	u0_dev = u_dev[idxs[0]]; u1_dev = u_dev[idxs[1]];
#endif

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
		out(u1:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&save_f);
#endif
#if defined(_OPENACC)
	get_time(&save_s);
	#pragma acc update host (u1[0:szarray])
	get_time(&save_f);
#endif
#if defined(__CUDACC__)
	get_time(&save_s);
	CUDA_SAFE_CALL(cudaMemcpy(u1, u1_dev, szarrayb, cudaMemcpyDeviceToHost));
	get_time(&save_f);
#endif
	double save_t = get_time_diff((struct timespec*)&save_s, (struct timespec*)&save_f);
	if (!no_timing) printf("data save time = %f sec (%f GB/sec)\n", save_t, szarrayb / (save_t * 1024 * 1024 * 1024));
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
		nocopy(u0:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(u1:length(szarray) alloc_if(0) free_if(1))
	{ }
	get_time(&free_f);
#endif
#if defined(__CUDACC__)
	get_time(&free_s);
	CUDA_SAFE_CALL(cudaFree(u0_dev));
	CUDA_SAFE_CALL(cudaFree(u1_dev));
	get_time(&free_f);
#endif
	double free_t = get_time_diff((struct timespec*)&free_s, (struct timespec*)&free_f);
	if (!no_timing) printf("device buffer free time = %f sec\n", free_t);
#endif

	// For the final mean - account only the norm of the top
	// most level (tracked by swapping idxs array of indexes).
	mean = 0.0f;
	for (int i = 0; i < szarray; i++)
		mean += u1[i];
	printf("final mean = %f\n", mean / szarray);

	free(u0);
	free(u1);

	fflush(stdout);

	return 0;
}

