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

#define _A(array, ix, iy, is) (array[(ix) + nx * (iy) + nx * ny * (is)])

// PATUS-generated kernel declaration.
#if defined(_PATUS)
void divergence_patus(real** dummy, real* u, real* ux, real* uy, real* uz,
	real alpha, real beta, real gamma, int nx, int ny, int ns);
#endif

#if defined(_MIC)
__attribute__((target(mic)))
#endif
#if defined(__CUDACC__)
extern "C" __global__
#endif
void divergence(int nx, int ny, int ns,
#if defined(__CUDACC__)
	kernelgen_cuda_config_t config,
#endif
	const real alpha, const real beta, const real gamma,
	real* u, real* ux, real* uy, real* uz)
{
#if defined(__CUDACC__)
	#define k_stride (config.strideDim.z)
	#define j_stride (config.strideDim.y)
	#define i_stride (config.strideDim.x)
	#define k_offset (blockIdx.z * blockDim.z + threadIdx.z)
	#define j_offset (blockIdx.y * blockDim.y + threadIdx.y)
	#define i_offset (blockIdx.x * blockDim.x + threadIdx.x)
	#define k_increment k_stride
	#define j_increment j_stride
	#define i_increment i_stride
#else
	#define k_offset 0
	#define j_offset 0
	#define i_offset 0
	#define k_increment 1
	#define j_increment 1
	#define i_increment 1
#endif
#if defined(_PATUS)
	real* dummy;
	#pragma omp parallel
	divergence_patus(&dummy, u, ux, uy, uz, alpha, beta, gamma, nx, ny, ns);
#else
#if defined(_OPENACC)
	size_t szarray = (size_t)nx * ny * ns;
	#pragma acc kernels loop independent gang(65535), present(u[0:szarray], ux[0:szarray], uy[0:szarray], uz[0:szarray])
#endif
#if defined(_OPENMP) || defined(_MIC)
	#pragma omp parallel for
#endif
	for (int k = 1 + k_offset; k < ns - 1; k += k_increment)
	{
#if defined(_OPENACC)
		#pragma acc loop independent
#endif
		for (int j = 1 + j_offset; j < ny - 1; j += j_increment)
		{
#if defined(_OPENACC)
			#pragma acc loop independent vector(512)
#endif
			for (int i = 1 + i_offset; i < nx - 1; i += i_increment)
			{
				_A(u, i, j, k) =
					alpha * (_A(ux, i+1, j, k) - _A(ux, i-1, j, k)) +
					beta  * (_A(uy, i, j+1, k) - _A(uy, i, j-1, k)) +
					gamma * (_A(uz, i, j, k+1) - _A(uz, i, j, k-1));
			}
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

	real alpha = real_rand();
	real beta  = real_rand();
	real gamma = real_rand();

	printf("alpha = %f, beta = %f, gamma = %f\n", alpha, beta, gamma);

	size_t szarray = (size_t)nx * ny * ns;
	size_t szarrayb = szarray * sizeof(real);

	real* u  = (real*)memalign(MEMALIGN, szarrayb);
	real* ux = (real*)memalign(MEMALIGN, szarrayb);
	real* uy = (real*)memalign(MEMALIGN, szarrayb);
	real* uz = (real*)memalign(MEMALIGN, szarrayb);

	if (!u || !ux || !uy || !uz)
	{
		printf("Error allocating memory for arrays: %p, %p, %p, %p\n", u, ux, uy, uz);
		exit(1);
	}

	real mean = 0.0f;
	for (int i = 0; i < szarray; i++)
	{
		u [i] = real_rand();
		ux[i] = real_rand();
		uy[i] = real_rand();
		uz[i] = real_rand();
		mean += u[i] + ux[i] + uy[i] + uz[i];
	}
	if (!no_timing) printf("initial mean = %f\n", mean / szarray / 4);

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
		nocopy(u:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(ux:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(uy:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(uz:length(szarray) alloc_if(0) free_if(0)),
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
		nocopy(u:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(ux:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(uy:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(uz:length(szarray) alloc_if(1) free_if(0))
	{ }
	get_time(&alloc_f);
#endif
#if defined(_OPENACC)
	get_time(&alloc_s);
	#pragma acc data create (u[0:szarray], ux[0:szarray], uy[0:szarray], uz[0:szarray])
	{
	get_time(&alloc_f);
#endif
#if defined(__CUDACC__)
	get_time(&alloc_s);
	real *u_dev = NULL, *ux_dev = NULL, *uy_dev = NULL, *uz_dev = NULL;
	CUDA_SAFE_CALL(cudaMalloc(&u_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&ux_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&uy_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&uz_dev, szarrayb));
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
		in(u:length(szarray) alloc_if(0) free_if(0)), \
		in(ux:length(szarray) alloc_if(0) free_if(0)), \
		in(uy:length(szarray) alloc_if(0) free_if(0)), \
		in(uz:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&load_f);
#endif
#if defined(_OPENACC)
	get_time(&load_s);
	#pragma acc update device(u[0:szarray], ux[0:szarray], uy[0:szarray], uz[0:szarray])
	get_time(&load_f);
#endif
#if defined(__CUDACC__)
	get_time(&load_s);
	CUDA_SAFE_CALL(cudaMemcpy(u_dev, u, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ux_dev, ux, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(uy_dev, uy, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(uz_dev, uz, szarrayb, cudaMemcpyHostToDevice));
	get_time(&load_f);
#endif
	double load_t = get_time_diff((struct timespec*)&load_s, (struct timespec*)&load_f);
	if (!no_timing) printf("data load time = %f sec (%f GB/sec)\n", load_t, 4 * szarrayb / (load_t * 1024 * 1024 * 1024));
#endif

	//
	// 4) Perform data processing iterations, keeping all data
	// on device.
	//
	volatile struct timespec compute_s, compute_f;
	get_time(&compute_s);
#if defined(_MIC)
	#pragma offload target(mic) \
		nocopy(u:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(ux:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(uy:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(uz:length(szarray) alloc_if(0) free_if(0))
#endif
#if defined(__CUDACC__)
	kernelgen_cuda_config_t config;
	kernelgen_cuda_configure_gird(1, nx, ny, ns, &config);
#endif
	{
		for (int it = 0; it < nt; it++)
		{
#if !defined(__CUDACC__)
			divergence(nx, ny, ns, alpha, beta, gamma, u, ux, uy, uz);
#else
			divergence<<<config.gridDim, config.blockDim, config.szshmem>>>(
				nx, ny, ns,
				config,
				alpha, beta, gamma, u_dev, ux_dev, uy_dev, uz_dev);
			CUDA_SAFE_CALL(cudaGetLastError());
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
		}
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
		out(u:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&save_f);
#endif
#if defined(_OPENACC)
	get_time(&save_s);
	#pragma acc update host (u[0:szarray])
	get_time(&save_f);
#endif
#if defined(__CUDACC__)
	get_time(&save_s);
	CUDA_SAFE_CALL(cudaMemcpy(u, u_dev, szarrayb, cudaMemcpyDeviceToHost));
	get_time(&save_f);
#endif
	double save_t = get_time_diff((struct timespec*)&save_s, (struct timespec*)&save_f);
	if (!no_timing) printf("data save time = %f sec (%f GB/sec)\n", save_t, szarrayb / (save_t * 1024 * 1024 * 1024));
#endif

	//
	// MIC or CUDA:
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
		nocopy(u:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(ux:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(uy:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(uz:length(szarray) alloc_if(0) free_if(1))
	{ }
	get_time(&free_f);
#endif
#if defined(__CUDACC__)
	get_time(&free_s);
	CUDA_SAFE_CALL(cudaFree(u_dev));
	CUDA_SAFE_CALL(cudaFree(ux_dev));
	CUDA_SAFE_CALL(cudaFree(uy_dev));
	CUDA_SAFE_CALL(cudaFree(uz_dev));
	get_time(&free_f);
#endif
	double free_t = get_time_diff((struct timespec*)&free_s, (struct timespec*)&free_f);
	if (!no_timing) printf("device buffer free time = %f sec\n", free_t);
#endif

	// For the final mean - account only the norm of the top
	// most level (tracked by swapping idxs array of indexes).
	mean = 0.0f;
	for (int i = 0; i < szarray; i++)
		mean += u[i];
	printf("final mean = %f\n", mean / szarray);

	free(u);
	free(ux);
	free(uy);
	free(uz);

	fflush(stdout);

	return 0;
}

