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

#if defined(_MIC)
__attribute__((target(mic)))
#endif
#if defined(__CUDACC__)
extern "C" __global__
#endif
void tricubic2(int nx, int ny, int ns,
#if defined(__CUDACC__)
	int i_stride, int j_stride, int k_stride,
#endif
	real* u0, real* u1, real* a, real* b, real* c)
{
#if defined(__CUDACC__)
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
#if defined(_OPENACC)
	size_t szarray = (size_t)nx * ny * ns;
	#pragma acc kernels loop independent gang(65535), present(u0[0:szarray], u1[0:szarray], a[0:szarray], b[0:szarray], c[0:szarray])
#endif
#if defined(_OPENMP) || defined(_MIC)
	#pragma omp parallel for
#endif
	for (int k = 2 + k_offset; k < ns - 2; k += k_increment)
	{
#if defined(_OPENACC)
		#pragma acc loop independent
#endif
		for (int j = 2 + j_offset; j < ny - 2; j += j_increment)
		{
#if defined(_OPENACC)
			#pragma acc loop independent vector(512)
#endif
			for (int i = 2 + i_offset; i < nx - 2; i += i_increment)
			{
				real wa[4], wb[4], wc[4];

				wa[0] =  1.0/6.0                      * _A(a, i,j,k) * (_A(a, i,j,k)+1.0) * (_A(a, i,j,k)+2.0);
				wa[1] = -0.5     * (_A(a, i,j,k)-1.0)                * (_A(a, i,j,k)+1.0) * (_A(a, i,j,k)+2.0);
				wa[2] =  0.5     * (_A(a, i,j,k)-1.0) * _A(a, i,j,k)                      * (_A(a, i,j,k)+2.0);
				wa[3] = -1.0/6.0 * (_A(a, i,j,k)-1.0) * _A(a, i,j,k) * (_A(a, i,j,k)+1.0);

				wb[0] =  1.0/6.0                      * _A(b, i,j,k) * (_A(b, i,j,k)+1.0) * (_A(b, i,j,k)+2.0);
				wb[1] = -0.5     * (_A(b, i,j,k)-1.0)                * (_A(b, i,j,k)+1.0) * (_A(b, i,j,k)+2.0);
				wb[2] =  0.5     * (_A(b, i,j,k)-1.0) * _A(b, i,j,k)                      * (_A(b, i,j,k)+2.0);
				wb[3] = -1.0/6.0 * (_A(b, i,j,k)-1.0) * _A(b, i,j,k) * (_A(b, i,j,k)+1.0);

				wc[0] =  1.0/6.0                      * _A(c, i,j,k) * (_A(c, i,j,k)+1.0) * (_A(c, i,j,k)+2.0);
				wc[1] = -0.5     * (_A(c, i,j,k)-1.0)                * (_A(c, i,j,k)+1.0) * (_A(c, i,j,k)+2.0);
				wc[2] =  0.5     * (_A(c, i,j,k)-1.0) * _A(c, i,j,k)                      * (_A(c, i,j,k)+2.0);
				wc[3] = -1.0/6.0 * (_A(c, i,j,k)-1.0) * _A(c, i,j,k) * (_A(c, i,j,k)+1.0);
	
				_A(u1, i,j,k) = 0;
				for (int kk = 0; kk < 4; kk++)
					for (int jj = 0; jj < 4; jj++)
						for (int ii = 0; ii < 4; ii++)
							_A(u1, i,j,k) += wa[ii] * wb[jj] * wc[kk] * _A(u0, i+ii-1,j+jj-1,k+kk-1);
			}
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

	size_t szarray = (size_t)nx * ny * ns;
	size_t szarrayb = szarray * sizeof(real);

	real* u0 = (real*)memalign(MEMALIGN, szarrayb);
	real* u1 = (real*)memalign(MEMALIGN, szarrayb);
	real* a = (real*)memalign(MEMALIGN, szarrayb);
	real* b = (real*)memalign(MEMALIGN, szarrayb);
	real* c = (real*)memalign(MEMALIGN, szarrayb);

	if (!u0 || !u1 || !a || !b || !c)
	{
		printf("Error allocating memory for arrays: %p, %p, %p, %p, %p\n", u0, u1, a, b, c);
		exit(1);
	}

	real mean = 0.0f;
	for (int i = 0; i < szarray; i++)
	{
		u0[i] = real_rand();
		u1[i] = real_rand();
		a[i] = real_rand();
		b[i] = real_rand();
		c[i] = real_rand();
		mean += u0[i] + u1[i] + a[i] + b[i] + c[i];
	}
	if (!no_timing) printf("initial mean = %f\n", mean / szarray / 5);

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
		nocopy(a:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(b:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(c:length(szarray) alloc_if(0) free_if(0))
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
		nocopy(a:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(b:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(c:length(szarray) alloc_if(1) free_if(0))
	{ }
	get_time(&alloc_f);
#endif
#if defined(_OPENACC)
	get_time(&alloc_s);
	#pragma acc data create (u0[0:szarray], u1[0:szarray], a[0:szarray], b[0:szarray], c[0:szarray])
	{
	get_time(&alloc_f);
#endif
#if defined(__CUDACC__)
	get_time(&alloc_s);
	float *u0_dev = NULL, *u1_dev = NULL, *a_dev = NULL, *b_dev = NULL, *c_dev = NULL;
	CUDA_SAFE_CALL(cudaMalloc(&u0_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&u1_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&a_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&b_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&c_dev, szarrayb));
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
		in(a:length(szarray) alloc_if(0) free_if(0)), \
		in(b:length(szarray) alloc_if(0) free_if(0)), \
		in(c:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&load_f);
#endif
#if defined(_OPENACC)
	get_time(&load_s);
	#pragma acc update device(u0[0:szarray], u1[0:szarray], a[0:szarray], b[0:szarray], c[0:szarray])
	get_time(&load_f);
#endif
#if defined(__CUDACC__)
	get_time(&load_s);
	CUDA_SAFE_CALL(cudaMemcpy(u0_dev, u0, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(u1_dev, u1, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(a_dev, a, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(b_dev, b, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(c_dev, c, szarrayb, cudaMemcpyHostToDevice));
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
		nocopy(a:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(b:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(c:length(szarray) alloc_if(0) free_if(0))
#endif
#if defined(__CUDACC__)
	dim3 gridDim, blockDim, strideDim;
	kernelgen_cuda_configure_gird(nx, ny, ns, &gridDim, &blockDim, &strideDim);
#endif
	{
		real *u0p = u0, *u1p = u1;

		for (int it = 0; it < nt; it++)
		{
#if !defined(__CUDACC__)
			tricubic2(nx, ny, ns, u0p, u1p, a, b, c);
#else
			tricubic2<<<gridDim, blockDim>>>(nx, ny, ns,
				strideDim.x, strideDim.y, strideDim.z,
				u0p, u1p, a, b, c);
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

	real* u[] = { u0, u1 }; 
	u0 = u[idxs[0]]; u1 = u[idxs[1]];

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
		nocopy(a:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(b:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(c:length(szarray) alloc_if(0) free_if(1))
	{ }
	get_time(&free_f);
#endif
#if defined(__CUDACC__)
	get_time(&free_s);
	CUDA_SAFE_CALL(cudaFree(u0_dev));
	CUDA_SAFE_CALL(cudaFree(u1_dev));
	CUDA_SAFE_CALL(cudaFree(a_dev));
	CUDA_SAFE_CALL(cudaFree(b_dev));
	CUDA_SAFE_CALL(cudaFree(c_dev));
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
	free(a);
	free(b);
	free(c);

	fflush(stdout);

	return 0;
}

