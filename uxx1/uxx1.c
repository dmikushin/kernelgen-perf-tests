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
void uxx1_patus(real** dummy, real* d1,
	real* u0, real* u1, real* xx, real* xy, real* xz,
	real dth, real c1, real c2, int nx, int ny, int ns);
#endif

#if defined(_MIC)
__attribute__((target(mic)))
#endif
#if defined(__CUDACC__)
extern "C" __global__
#endif
void uxx1(int nx, int ny, int ns,
#if defined(__CUDACC__)
	kernelgen_cuda_config_t config,
#endif
	const real c1, const real c2,
	real* u0, real* u1, real* d1, real* xx, real* xy, real* xz)
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
	real dth = 1. / nx;
#if defined(_PATUS)
	real* dummy;
	#pragma omp parallel
	uxx1_patus(&dummy, d1, u0, u1, xx, xy, xz, dth, c1, c2, nx, ny, ns);
#else
#if defined(_OPENACC)
	size_t szarray = (size_t)nx * ny * ns;
	#pragma acc kernels loop independent gang(65535), present(u0[0:szarray], u1[0:szarray], d1[0:szarray], xx[0:szarray], xy[0:szarray], xz[0:szarray])
#endif
#if defined(_OPENMP) || defined(_MIC)
	#pragma omp parallel for
#endif
	for (int k = 2 + k_offset; k < ns - 1; k += k_increment)
	{
#if defined(_OPENACC)
		#pragma acc loop independent
#endif
		for (int j = 2 + j_offset; j < ny - 1; j += j_increment)
		{
#if defined(_OPENACC)
			#pragma acc loop independent vector(512)
#endif
			for (int i = 2 + i_offset; i < nx - 1; i += i_increment)
			{
				real d = 0.25 * (
					_A(d1, i, j, k  ) + _A(d1, i, j-1, k  ) +
					_A(d1, i, j, k-1) + _A(d1, i, j-1, k-1));
				
				_A(u1, i, j, k) = _A(u0, i, j, k) + (dth / d) * (
					c1 * (_A(xx, i  , j  , k  ) - _A(xx, i-1, j  , k  )) +
					c2 * (_A(xx, i+1, j  , k  ) - _A(xx, i-2, j  , k  )) +
					c1 * (_A(xy, i  , j  , k  ) - _A(xy, i  , j-1, k  )) +
					c2 * (_A(xy, i  , j+1, k  ) - _A(xy, i  , j-2, k  )) +
					c1 * (_A(xz, i  , j  , k  ) - _A(xz, i  , j  , k-1)) +
					c2 * (_A(xz, i  , j  , k+1) - _A(xz, i  , j  , k-2)));
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

	real c1 = real_rand();
	real c2 = real_rand();

	printf("c1 = %f, c2 = %f\n", c1, c2);

	size_t szarray = (size_t)nx * ny * ns;
	size_t szarrayb = szarray * sizeof(real);

	real* u0 = (real*)memalign(MEMALIGN, szarrayb);
	real* u1 = (real*)memalign(MEMALIGN, szarrayb);
	real* d1 = (real*)memalign(MEMALIGN, szarrayb);
	real* xx = (real*)memalign(MEMALIGN, szarrayb);
	real* xy = (real*)memalign(MEMALIGN, szarrayb);
	real* xz = (real*)memalign(MEMALIGN, szarrayb);

	if (!u0 || !u1 || !d1 || !xx || !xy || !xz)
	{
		printf("Error allocating memory for arrays: %p, %p, %p, %p, %p, %p\n",
			u0, u1, d1, xx, xy, xz);
		exit(1);
	}

	real mean = 0.0f;
	for (int i = 0; i < szarray; i++)
	{
		u0[i] = real_rand();
		u1[i] = real_rand();
		d1[i] = real_rand();
		xx[i] = real_rand();
		xy[i] = real_rand();
		xz[i] = real_rand();
		mean += u0[i] + u1[i] + d1[i] + xx[i] + xy[i] + xz[i];
	}
	if (!no_timing) printf("initial mean = %f\n", mean / szarray / 6);

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
		nocopy(u1:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(d1:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(xx:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(xy:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(xz:length(szarray) alloc_if(0) free_if(0))
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
		nocopy(u1:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(d1:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(xx:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(xy:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(xz:length(szarray) alloc_if(1) free_if(0))
	{ }
	get_time(&alloc_f);
#endif
#if defined(_OPENACC)
	get_time(&alloc_s);
	#pragma acc data create (u0[0:szarray], u1[0:szarray], d1[0:szarray], xx[0:szarray], xy[0:szarray], xz[0:szarray])
	{
	get_time(&alloc_f);
#endif
#if defined(__CUDACC__)
	get_time(&alloc_s);
	real *u0_dev = NULL, *u1_dev = NULL, *d1_dev = NULL;
	real *xx_dev = NULL, *xy_dev = NULL, *xz_dev = NULL;
	CUDA_SAFE_CALL(cudaMalloc(&u0_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&u1_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&d1_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&xx_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&xy_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&xz_dev, szarrayb));
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
		in(u1:length(szarray) alloc_if(0) free_if(0)), \
		in(d1:length(szarray) alloc_if(0) free_if(0)), \
		in(xx:length(szarray) alloc_if(0) free_if(0)), \
		in(xy:length(szarray) alloc_if(0) free_if(0)), \
		in(xz:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&load_f);
#endif
#if defined(_OPENACC)
	get_time(&load_s);
	#pragma acc update device(u0[0:szarray], u1[0:szarray], d1[0:szarray], xx[0:szarray], xy[0:szarray], xz[0:szarray])
	get_time(&load_f);
#endif
#if defined(__CUDACC__)
	get_time(&load_s);
	CUDA_SAFE_CALL(cudaMemcpy(u0_dev, u0, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(u1_dev, u1, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d1_dev, d1, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(xx_dev, xx, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(xy_dev, xy, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(xz_dev, xz, szarrayb, cudaMemcpyHostToDevice));
	get_time(&load_f);
#endif
	double load_t = get_time_diff((struct timespec*)&load_s, (struct timespec*)&load_f);
	if (!no_timing) printf("data load time = %f sec (%f GB/sec)\n", load_t, 5 * szarrayb / (load_t * 1024 * 1024 * 1024));
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
		nocopy(u1:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(d1:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(xx:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(xy:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(xz:length(szarray) alloc_if(0) free_if(0))
#endif
#if defined(__CUDACC__)
	kernelgen_cuda_config_t config;
	kernelgen_cuda_configure_gird(1, nx, ny, ns, &config);
#endif
	{
#if !defined(__CUDACC__)
		real *u0p = u0, *u1p = u1;
		real *d1p = d1, *xxp = xx, *xyp = xy, *xzp = xz;
#else
		real *u0p = u0_dev, *u1p = u1_dev;
		real *d1p = d1_dev, *xxp = xx_dev, *xyp = xy_dev, *xzp = xz_dev;
#endif
		for (int it = 0; it < nt; it++)
		{
#if !defined(__CUDACC__)
			uxx1(nx, ny, ns, c1, c2, u0p, u1p, d1, xx, xy, xz);
#else
			uxx1<<<config.gridDim, config.blockDim, config.szshmem>>>(
				nx, ny, ns,
				config,
				c1, c2, u0p, u1p, d1p, xxp, xyp, xzp);
			CUDA_SAFE_CALL(cudaGetLastError());
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
			real* u = u0p; u0p = u1p; u1p = u;
			int idx = idxs[0]; idxs[0] = idxs[1]; idxs[1] = idx;
		}
	}
	get_time(&compute_f);
	double compute_t = get_time_diff((struct timespec*)&compute_s, (struct timespec*)&compute_f);
	if (!no_timing) printf("compute time = %f sec\n", compute_t);

#if !defined(__CUDACC__)
	real* u[] = { u0, u1 }; 
	u0 = u[idxs[0]]; u1 = u[idxs[1]];
#else
	real* u[] = { u0_dev, u1_dev }; 
	u0_dev = u[idxs[0]]; u1_dev = u[idxs[1]];
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
		nocopy(u1:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(d1:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(xx:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(xy:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(xz:length(szarray) alloc_if(0) free_if(1))
	{ }
	get_time(&free_f);
#endif
#if defined(__CUDACC__)
	get_time(&free_s);
	CUDA_SAFE_CALL(cudaFree(u0_dev));
	CUDA_SAFE_CALL(cudaFree(u1_dev));
	CUDA_SAFE_CALL(cudaFree(d1_dev));
	CUDA_SAFE_CALL(cudaFree(xx_dev));
	CUDA_SAFE_CALL(cudaFree(xy_dev));
	CUDA_SAFE_CALL(cudaFree(xz_dev));
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
	free(d1);
	free(xx);
	free(xy);
	free(xz);

	fflush(stdout);

	return 0;
}

