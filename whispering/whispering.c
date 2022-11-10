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
#define _A3(array, ix, iy, is) (array[(ix) + nx * (iy) + nx * ny * (is)])

// PATUS-generated kernel declaration.
#if defined(_PATUS)
void whispering_patus(
	real** dummy1, real** dummy2, real** dummy3, real** dummy4,
	real* e, real* h0, real* h1, real* u_em0, real* u_em1,
	real* ca, real* cb, real* da, real* db,
	real mu, real epsilon, int nx, int ny);
#endif

#if defined(_MIC)
__attribute__((target(mic)))
#endif
#if defined(__CUDACC__)
extern "C" __global__
#endif
void whispering(int nx, int ny,
#if defined(__CUDACC__)
	kernelgen_cuda_config_t config,
#endif
	const real mu, const real epsilon,
	real* e, real* h0, real* h1, real* u_em0, real* u_em1,
	real* ca, real* cb, real* da, real* db)
{
#if defined(__CUDACC__)
	#define j_stride (config.strideDim.y)
	#define i_stride (config.strideDim.x)
	#define j_offset (blockIdx.y * blockDim.y + threadIdx.y)
	#define i_offset (blockIdx.x * blockDim.x + threadIdx.x)
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
	real *dummy1, *dummy2, *dummy3, *dummy4;
	#pragma omp parallel
	whispering_patus(&dummy1, &dummy2, &dummy3, &dummy4,
		e, h0, h1, u_em0, u_em1,
		ca, cb, da, db, mu, epsilon, nx, ny);
#else
#if defined(_OPENACC)
	size_t szarray = (size_t)nx * ny;
	#pragma acc kernels loop independent gang(65535), present(e[0:2 * szarray], \
		e[0:2 * szarray], h0[0:szarray], h1[0:szarray], u_em0[0:szarray], u_em1[0:szarray], \
		ca[0:szarray], cb[0:szarray], da[0:szarray], db[0:szarray])
#endif
#if defined(_OPENMP) || defined(_MIC)
	#pragma omp parallel for
#endif
	for (int j = 1 + j_offset; j < ny - 1; j += j_increment)
	{
#if defined(_OPENACC)
		#pragma acc loop independent vector(512)
#endif
		for (int i = 1 + i_offset; i < nx - 1; i += i_increment)
		{
			real e0 = _A(ca, i, j) * _A3(e, i, j, 0) + _A(cb, i, j) * (_A(h0, i, j+1) - _A(h0, i, j));
			real e1 = _A(ca, i, j) * _A3(e, i, j, 1) - _A(cb, i, j) * (_A(h0, i+1, j) - _A(h0, i, j));
        
			real ey = _A(ca, i, j-1) * _A3(e, i, j-1, 0) + _A(cb, i, j-1) * (_A(h0, i, j) - _A(h0, i, j-1));
			real ex = _A(ca, i-1, j) * _A3(e, i-1, j, 1) - _A(cb, i-1, j) * (_A(h0, i, j) - _A(h0, i-1, j));

			_A3(e, i, j, 0) = e0;
			_A3(e, i, j, 1) = e1;

			real h = _A(da, i, j) * _A(h0, i, j) + _A(db, i, j) * (e0 - ey + ex - e1);
			_A(h1, i, j) = h;
			
			_A(u_em1, i, j) = _A(u_em0, i, j) + 0.5 * (h * h / mu + epsilon * (e0 * e0 + e1 * e1));
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

	real mu = real_rand();
	real epsilon = real_rand();

	printf("mu = %f, epsilon = %f\n", mu, epsilon);

	size_t szarray = (size_t)nx * ny;
	size_t szarrayb = szarray * sizeof(real);

	real* e = (real*)memalign(MEMALIGN, 2 * szarrayb);
	real* h0 = (real*)memalign(MEMALIGN, szarrayb);
	real* h1 = (real*)memalign(MEMALIGN, szarrayb);
	real* u_em0 = (real*)memalign(MEMALIGN, szarrayb);
	real* u_em1 = (real*)memalign(MEMALIGN, szarrayb);
	real* ca = (real*)memalign(MEMALIGN, szarrayb);
	real* cb = (real*)memalign(MEMALIGN, szarrayb);
	real* da = (real*)memalign(MEMALIGN, szarrayb);
	real* db = (real*)memalign(MEMALIGN, szarrayb);

	if (!e || !h0 || !h1 || !u_em0 || !u_em1 || !ca || !cb || !da || !db)
	{
		printf("Error allocating memory for arrays: %p, %p, %p, %p, %p, %p, %p, %p, %p\n",
			e, h0, h1, u_em0, u_em1, ca, cb, da, db);
		exit(1);
	}

	real mean = 0.0f;
	for (int i = 0; i < szarray; i++)
	{
		e[2 * i] = real_rand();
		e[2 * i + 1] = real_rand();
		h0[i] = real_rand();
		h1[i] = real_rand();
		u_em0[i] = real_rand();
		u_em1[i] = real_rand();
		ca[i] = real_rand();
		cb[i] = real_rand();
		da[i] = real_rand();
		db[i] = real_rand();
		mean += e[2 * i] + e[2 * i + 1] + h0[i] + h1[i] + u_em0[i] + u_em1[i] + ca[i] + cb[i] + da[i] + db[i];
	}
	printf("initial mean = %f\n", mean / szarray / 12);

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
		nocopy(e:length(2 * szarray) alloc_if(0) free_if(0)), \
		nocopy(h0:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(h1:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(u_em0:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(u_em1:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(ca:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(cb:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(da:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(db:length(szarray) alloc_if(0) free_if(0))
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
		nocopy(e:length(2 * szarray) alloc_if(1) free_if(0)), \
		nocopy(h0:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(h1:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(u_em0:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(u_em1:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(ca:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(cb:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(da:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(db:length(szarray) alloc_if(1) free_if(0))
	{ }
	get_time(&alloc_f);
#endif
#if defined(_OPENACC)
	get_time(&alloc_s);
	#pragma acc data create (e[0:2 * szarray], h0[0:szarray], h1[0:szarray], u_em0[0:szarray], u_em1[0:szarray], ca[0:szarray], cb[0:szarray], da[0:szarray], db[0:szarray])
	{
	get_time(&alloc_f);
#endif
#if defined(__CUDACC__)
	get_time(&alloc_s);
	real *e_dev = NULL, *h0_dev = NULL, *h1_dev = NULL;
	real *u_em0_dev = NULL, *u_em1_dev = NULL, *ca_dev = NULL, *cb_dev = NULL;
	real *da_dev = NULL, *db_dev = NULL;
	CUDA_SAFE_CALL(cudaMalloc(&e_dev, 2 * szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&h0_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&h1_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&u_em0_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&u_em1_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&ca_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&cb_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&da_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&db_dev, szarrayb));
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
		in(e:length(2 * szarray) alloc_if(0) free_if(0)), \
		in(h0:length(szarray) alloc_if(0) free_if(0)), \
		in(h1:length(szarray) alloc_if(0) free_if(0)), \
		in(u_em0:length(szarray) alloc_if(0) free_if(0)), \
		in(u_em1:length(szarray) alloc_if(0) free_if(0)), \
		in(ca:length(szarray) alloc_if(0) free_if(0)), \
		in(cb:length(szarray) alloc_if(0) free_if(0)), \
		in(da:length(szarray) alloc_if(0) free_if(0)), \
		in(db:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&load_f);
#endif
#if defined(_OPENACC)
	get_time(&load_s);
	#pragma acc update device(e[0:2 * szarray], h0[0:szarray], h1[0:szarray], u_em0[0:szarray], u_em1[0:szarray], ca[0:szarray], cb[0:szarray], da[0:szarray], db[0:szarray])
	get_time(&load_f);
#endif
#if defined(__CUDACC__)
	get_time(&load_s);
	CUDA_SAFE_CALL(cudaMemcpy(e_dev, e, 2 * szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(h0_dev, h0, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(h1_dev, h1, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(u_em0_dev, u_em0, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(u_em1_dev, u_em1, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(ca_dev, ca, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(cb_dev, cb, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(da_dev, da, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(db_dev, db, szarrayb, cudaMemcpyHostToDevice));
	get_time(&load_f);
#endif
	double load_t = get_time_diff((struct timespec*)&load_s, (struct timespec*)&load_f);
	if (!no_timing) printf("data load time = %f sec (%f GB/sec)\n", load_t, 10 * szarrayb / (load_t * 1024 * 1024 * 1024));
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
		nocopy(e:length(2 * szarray) alloc_if(0) free_if(0)), \
		nocopy(h0:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(h1:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(u_em0:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(u_em1:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(ca:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(cb:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(da:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(db:length(szarray) alloc_if(0) free_if(0))
#endif
#if defined(__CUDACC__)
	kernelgen_cuda_config_t config;
	kernelgen_cuda_configure_gird(1, nx, ny, 1, &config);
#endif
	{
#if !defined(__CUDACC__) || defined(_PPCG)
		real *h0p = h0, *h1p = h1;
		real *u_em0p = u_em0, *u_em1p = u_em1;
#else
		real *h0p = h0_dev, *h1p = h1_dev;
		real *u_em0p = u_em0_dev, *u_em1p = u_em1_dev;
#endif
		for (int it = 0; it < nt; it++)
		{
#if !defined(__CUDACC__)
			whispering(nx, ny, mu, epsilon, e, h0p, h1p, u_em0p, u_em1p, ca, cb, da, db);
#else
			whispering<<<config.gridDim, config.blockDim, config.szshmem>>>(
				nx, ny,
				config,
				mu, epsilon, e_dev, h0p, h1p, u_em0p, u_em1p, ca_dev, cb_dev, da_dev, db_dev);
			CUDA_SAFE_CALL(cudaGetLastError());
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
			real* h = h0p; h0p = h1p; h1p = h;
			real* u_em = u_em0p; u_em0p = u_em1p; u_em1p = u_em;
			int idx = idxs[0]; idxs[0] = idxs[1]; idxs[1] = idx;
		}
	}
	get_time(&compute_f);
	double compute_t = get_time_diff((struct timespec*)&compute_s, (struct timespec*)&compute_f);
	if (!no_timing) printf("compute time = %f sec\n", compute_t);

	real* h[] = { h0, h1 };
	h0 = h[idxs[0]]; h1 = h[idxs[1]];
	real* u_em[] = { u_em0, u_em1 };
	u_em0 = u_em[idxs[0]]; u_em1 = u_em[idxs[1]];

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
		out(e:length(2 * szarray) alloc_if(0) free_if(0)), \
		out(h0:length(szarray) alloc_if(0) free_if(0)), \
		out(h1:length(szarray) alloc_if(0) free_if(0)), \
		out(u_em0:length(szarray) alloc_if(0) free_if(0)), \
		out(u_em1:length(szarray) alloc_if(0) free_if(0)), \
		out(ca:length(szarray) alloc_if(0) free_if(0)), \
		out(cb:length(szarray) alloc_if(0) free_if(0)), \
		out(da:length(szarray) alloc_if(0) free_if(0)), \
		out(db:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&save_f);
#endif
#if defined(_OPENACC)
	get_time(&save_s);
	#pragma acc update host (e[0:2 * szarray], h0[0:szarray], h1[0:szarray], u_em0[0:szarray], u_em1[0:szarray], ca[0:szarray], cb[0:szarray], da[0:szarray], db[0:szarray])
	get_time(&save_f);
#endif
#if defined(__CUDACC__)
	get_time(&save_s);
	CUDA_SAFE_CALL(cudaMemcpy(e, e_dev, 2 * szarrayb, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h0, h0_dev, szarrayb, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h1, h1_dev, szarrayb, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(u_em0, u_em0_dev, szarrayb, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(u_em1, u_em1_dev, szarrayb, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(ca, ca_dev, szarrayb, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(cb, cb_dev, szarrayb, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(da, da_dev, szarrayb, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(db, db_dev, szarrayb, cudaMemcpyDeviceToHost));
	get_time(&save_f);
#endif
	double save_t = get_time_diff((struct timespec*)&save_s, (struct timespec*)&save_f);
	if (!no_timing) printf("data save time = %f sec (%f GB/sec)\n", save_t, 10 * szarrayb / (save_t * 1024 * 1024 * 1024));
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
		nocopy(e:length(2 * szarray) alloc_if(0) free_if(1)), \
		nocopy(h0:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(h1:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(u_em0:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(u_em1:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(ca:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(cb:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(da:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(db:length(szarray) alloc_if(0) free_if(1))
	{ }
	get_time(&free_f);
#endif
#if defined(__CUDACC__)
	get_time(&free_s);
	CUDA_SAFE_CALL(cudaFree(e_dev));
	CUDA_SAFE_CALL(cudaFree(h0_dev));
	CUDA_SAFE_CALL(cudaFree(h1_dev));
	CUDA_SAFE_CALL(cudaFree(u_em0_dev));
	CUDA_SAFE_CALL(cudaFree(u_em1_dev));
	CUDA_SAFE_CALL(cudaFree(ca_dev));
	CUDA_SAFE_CALL(cudaFree(cb_dev));
	CUDA_SAFE_CALL(cudaFree(da_dev));
	CUDA_SAFE_CALL(cudaFree(db_dev));
	get_time(&free_f);
#endif
	double free_t = get_time_diff((struct timespec*)&free_s, (struct timespec*)&free_f);
	if (!no_timing) printf("device buffer free time = %f sec\n", free_t);
#endif

	// For the final mean - account only the norm of the top
	// most level (tracked by swapping idxs array of indexes).
	mean = 0.0f;
	for (int i = 0; i < szarray; i++)
	{
		mean += e[2 * i] + e[2 * i + 1] + h1[i] + u_em1[i];
	}
	printf("final mean = %f\n", mean / szarray / 4);

	free(e);
	free(h0);
	free(h1);
	free(u_em0);
	free(u_em1);
	free(ca);
	free(cb);
	free(da);
	free(db);

	fflush(stdout);

	return 0;
}

