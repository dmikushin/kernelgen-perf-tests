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

#if defined(__CUDACC__)
extern "C" __global__
#endif
void matmul_(
#if defined(__CUDACC__)
	int nx, int ny, int ns,
	int i_stride, int j_stride,
#else
	int* nx, int* ny, int* ns,
#endif
	real* A, real* B, real* C);

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
	if (argc != 5)
	{
		printf("Usage: %s <nx> <ny> <ns> <nt>\n", argv[0]);
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
	parse_arg(ns, argv[3]);
	parse_arg(nt, argv[4]);

	size_t szarrayA = (size_t)nx * ny;
	size_t szarrayAb = szarrayA * sizeof(real);
	size_t szarrayB = (size_t)ny * ns;
	size_t szarrayBb = szarrayB * sizeof(real);
	size_t szarrayC = (size_t)nx * ns;
	size_t szarrayCb = szarrayC * sizeof(real);

	real* A = (real*)memalign(MEMALIGN, szarrayAb);
	real* B = (real*)memalign(MEMALIGN, szarrayBb);
	real* C = (real*)memalign(MEMALIGN, szarrayCb);

	if (!A || !B || !C)
	{
		printf("Error allocating memory for arrays: %p, %p, %p\n", A, B, C);
		exit(1);
	}

	real meanA = 0.0f;
	for (int i = 0; i < szarrayA; i++)
	{
		A[i] = real_rand();
		meanA += A[i];
	}
	real meanB = 0.0f;
	for (int i = 0; i < szarrayB; i++)
	{
		B[i] = real_rand();
		meanB += B[i];
	}
	printf("initial mean = %f\n", (meanA / szarrayA + meanB / szarrayB));

	//
	// MIC or OPENACC:
	//
	// 1) Perform an empty offload, that should strip
	// the initialization time from further offloads.
	//
#if defined(_MIC) || defined(_OPENACC) || defined(__CUDACC__)
	volatile struct timespec init_s, init_f;
#if defined(_MIC)
	get_time(&init_s);
	#pragma offload target(mic) \
		nocopy(A:length(szarrayA) alloc_if(0) free_if(0)), \
		nocopy(B:length(szarrayB) alloc_if(0) free_if(0)), \
		nocopy(C:length(szarrayC) alloc_if(0) free_if(0))
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
	// MIC or OPENACC:
	//
	// 2) Allocate data on device, but do not copy anything.
	//
#if defined(_MIC) || defined(_OPENACC) || defined(__CUDACC__)
	volatile struct timespec alloc_s, alloc_f;
#if defined(_MIC)
	get_time(&alloc_s);
	#pragma offload target(mic) \
		nocopy(A:length(szarrayA) alloc_if(1) free_if(0)), \
		nocopy(B:length(szarrayB) alloc_if(1) free_if(0)), \
		nocopy(C:length(szarrayC) alloc_if(1) free_if(0))
	{ }
	get_time(&alloc_f);
#endif
#if defined(_OPENACC)
	get_time(&alloc_s);
	#pragma acc data create (A[0:szarrayA], B[0:szarrayB], C[0:szarrayC])
	{
	get_time(&alloc_f);
#endif
#if defined(__CUDACC__)
	get_time(&alloc_s);
	real *A_dev = NULL, *B_dev = NULL, *C_dev = NULL;
	CUDA_SAFE_CALL(cudaMalloc(&A_dev, szarrayAb));
	CUDA_SAFE_CALL(cudaMalloc(&B_dev, szarrayBb));
	CUDA_SAFE_CALL(cudaMalloc(&C_dev, szarrayCb));
	get_time(&alloc_f);
#endif
	double alloc_t = get_time_diff((struct timespec*)&alloc_s, (struct timespec*)&alloc_f);
	if (!no_timing) printf("device buffer alloc time = %f sec\n", alloc_t);
#endif

	//
	// MIC or OPENACC:
	//
	// 3) Transfer data from host to device and leave it there,
	// i.e. do not allocate deivce memory buffers.
	//
#if defined(_MIC) || defined(_OPENACC) || defined(__CUDACC__)
	volatile struct timespec load_s, load_f;
#if defined(_MIC)
	get_time(&load_s);
	#pragma offload target(mic) \
		in(A:length(szarrayA) alloc_if(0) free_if(0)), \
		in(B:length(szarrayB) alloc_if(0) free_if(0)), \
		in(C:length(szarrayC) alloc_if(0) free_if(0))
	{ }
	get_time(&load_f);
#endif
#if defined(_OPENACC)
	get_time(&load_s);
	#pragma acc update device(A[0:szarrayA], B[0:szarrayB], C[0:szarrayC])
	get_time(&load_f);
#endif
#if defined(__CUDACC__)
	get_time(&load_s);
	CUDA_SAFE_CALL(cudaMemcpy(A_dev, A, szarrayAb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(B_dev, B, szarrayBb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(C_dev, C, szarrayCb, cudaMemcpyHostToDevice));
	get_time(&load_f);
#endif
	double load_t = get_time_diff((struct timespec*)&load_s, (struct timespec*)&load_f);
	if (!no_timing) printf("data load time = %f sec (%f GB/sec)\n", load_t, (szarrayAb + szarrayBb) / (load_t * 1024 * 1024 * 1024));
#endif

	//
	// 4) Perform data processing iterations, keeping all data
	// on device.
	//
	volatile struct timespec compute_s, compute_f;
	get_time(&compute_s);
#if defined(_MIC)
	#pragma offload target(mic) \
		nocopy(A:length(szarrayA) alloc_if(0) free_if(0)), \
		nocopy(B:length(szarrayB) alloc_if(0) free_if(0)), \
		nocopy(C:length(szarrayC) alloc_if(0) free_if(0))
#endif
#if defined(__CUDACC__)
	kernelgen_cuda_config_t config;
	kernelgen_cuda_configure_gird(1, nx, ny, ns, &config);
#endif
	{
		for (int it = 0; it < nt; it++)
		{
#if !defined(__CUDACC__)
			matmul_(&nx, &ny, &ns, A, B, C);
#else
			matmul_<<<config.gridDim, config.blockDim, config.szshmem>>>(
				nx, ny, ns,
				config.strideDim.x, config.strideDim.y,
				A_dev, B_dev, C_dev);
			CUDA_SAFE_CALL(cudaGetLastError());
			CUDA_SAFE_CALL(cudaStreamSynchronize(0));
#endif
		}
	}
	get_time(&compute_f);
	double compute_t = get_time_diff((struct timespec*)&compute_s, (struct timespec*)&compute_f);
	if (!no_timing) printf("compute time = %f sec\n", compute_t);

	//
	// MIC or OPENACC:
	//
	// 5) Transfer output data back from device to host.
	//
#if defined(_MIC) || defined(_OPENACC) || defined(__CUDACC__)
	volatile struct timespec save_s, save_f;
#if defined(_MIC)
	get_time(&save_s);
	#pragma offload target(mic) \
		out(C:length(szarrayC) alloc_if(0) free_if(0))
	{ }
	get_time(&save_f);
#endif
#if defined(_OPENACC)
	get_time(&save_s);
	#pragma acc update host (C[0:szarrayC])
	get_time(&save_f);
#endif
#if defined(__CUDACC__)
	get_time(&save_s);
	CUDA_SAFE_CALL(cudaMemcpy(C, C_dev, szarrayCb, cudaMemcpyDeviceToHost));
	get_time(&save_f);
#endif
	double save_t = get_time_diff((struct timespec*)&save_s, (struct timespec*)&save_f);
	if (!no_timing) printf("data save time = %f sec (%f GB/sec)\n", save_t, szarrayCb / (save_t * 1024 * 1024 * 1024));
#endif

	//
	// MIC or OPENACC:
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
		nocopy(A:length(szarrayA) alloc_if(0) free_if(1)), \
		nocopy(B:length(szarrayB) alloc_if(0) free_if(1)), \
		nocopy(C:length(szarrayC) alloc_if(0) free_if(1))
	{ }
	get_time(&free_f);
#endif
#if defined(__CUDACC__)
	get_time(&free_s);
	CUDA_SAFE_CALL(cudaFree(A_dev));
	CUDA_SAFE_CALL(cudaFree(B_dev));
	CUDA_SAFE_CALL(cudaFree(C_dev));
	get_time(&free_f);
#endif
	double free_t = get_time_diff((struct timespec*)&free_s, (struct timespec*)&free_f);
	if (!no_timing) printf("device buffer free time = %f sec\n", free_t);
#endif

	// For the final mean - account only the norm of the top
	// most level (tracked by swapping idxs array of indexes).
	real meanC = 0.0f;
	for (int i = 0; i < szarrayC; i++)
		meanC += C[i];
	printf("final mean = %f\n", meanC / szarrayC);

	free(A);
	free(B);
	free(C);

	fflush(stdout);

	return 0;
}

