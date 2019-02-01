// TODO parameterize shadow boundaries thicknesses

//===----------------------------------------------------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
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

// Stencil boundaries thicknesses used to configure CUDA shared memory.
// In stencil code itself we still use plane numbers for better readability.
#if defined(__CUDA_SHMEM1D__) || defined(__CUDA_SHMEM2D__)
#define STENCIL_BOUNDARY_LEFT 2
#define STENCIL_BOUNDARY_RIGHT 2 
#define STENCIL_BOUNDARY_BOTTOM 2
#define STENCIL_BOUNDARY_TOP 2
#define STENCIL_BOUNDARY_BACK 2
#define STENCIL_BOUNDARY_FRONT 2
#define STENCIL_BANK_SHIFT 2
#endif

#if defined(__CUDA_VECTORIZE2__)
#define _A(array, is, iy, ix) (((real2*)(&array[(ix) + nx * (iy) + nx * ny * (is)]))[0])
#define _AS(array, is, iy, ix) (array[(ix) + nx * (iy) + nx * ny * (is)])
#else
#if !defined(__CUDACC__) && defined(_PPCG)
#define _A(array, is, iy, ix) (array[is][iy][ix])
#else
#define _A(array, is, iy, ix) (array[(ix) + nx * (iy) + nx * ny * (is)])
#endif
#endif

#if defined(__CUDA_VECTORIZE2__)
#define _S1D(array, ix) (((real2*)(&array##_shm[(ix)]))[0])
#define _S2D(array, iy, ix) (((real2*)(&array##_shm[(int)(ix) + 2 * (int)(iy) * ((int)blockDim.x + \
	STENCIL_BOUNDARY_LEFT + STENCIL_BOUNDARY_RIGHT + STENCIL_BANK_SHIFT)]))[0])
#else
#define _S1D(array, ix) (array##_shm[(ix)])
#define _S2D(array, iy, ix) (array##_shm[(int)(ix) + (int)(iy) * ((int)blockDim.x + \
	STENCIL_BOUNDARY_LEFT + STENCIL_BOUNDARY_RIGHT + STENCIL_BANK_SHIFT)])
#endif

#define _R1D(array, ix) (array##_reg[(ix)])

// PATUS-generated kernel declaration.
#if defined(_PATUS)
void wave13pt_patus(real** dummy, real* w0, real* w1, real* w2,
	real m0, real m1, real m2, int nx, int ny, int ns);
#endif

#if defined(_MIC)
__attribute__((target(mic)))
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
extern "C" __global__
#endif
void wave13pt(const int nx, const int ny, const int ns,
#if defined(__CUDACC__) && !defined(_PPCG)
	kernelgen_cuda_config_t config,
#endif
	const real m0, const real m1, const real m2,
	const real* const __restrict__ w0p, const real* const __restrict__ w1p,
	real* const __restrict__ w2p)
{
#if !defined(__CUDACC__) && defined(_PPCG)
	real (*w0)[ny][nx] = (real(*)[ny][nx])w0p;
	real (*w1)[ny][nx] = (real(*)[ny][nx])w1p;
	real (*w2)[ny][nx] = (real(*)[ny][nx])w2p;
#else
	const real* const __restrict__ w0 = w0p;
	const real* const __restrict__ w1 = w1p;
	real* const __restrict__ w2 = w2p;
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
	#define i_stride (config.strideDim.x)
	#define j_stride (config.strideDim.y)
	#define k_stride (config.strideDim.z)
#if defined(__CUDA_SHMEM1D__) || defined(__CUDA_SHMEM2D__)
	extern __shared__ char shmem[];
	real* w1_shm = (real*)(shmem + config.shmem_arrays[0]);
#endif
#if defined(__CUDA_SHUFFLE__)
	uint32_t laneid;
	asm("mov.u32 %0, %laneid;" : "=r"(laneid));
#endif
	#define k_offset (blockIdx.z * blockDim.z + threadIdx.z)
	#define j_offset (blockIdx.y * blockDim.y + threadIdx.y)
#if defined(__CUDA_VECTORIZE2__)
	#define i_offset ((blockIdx.x * blockDim.x + threadIdx.x) * 2)
#else
	#define i_offset (blockIdx.x * blockDim.x + threadIdx.x)
#endif
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
	wave13pt_patus(&dummy, w0, w1, w2, m0, m1, m2, nx, ny, ns);
#else
#if defined(_OPENACC)
	size_t szarray = (size_t)nx * ny * ns;
	#pragma acc kernels loop independent gang(ns), present(w0[0:szarray], w1[0:szarray], w2[0:szarray])
#endif
#if defined(_OPENMP) || defined(_MIC)
	#pragma omp parallel for
#endif
#if defined(_PPCG)
	#pragma scop
#endif
#if !defined(__CUDA_SHMEM2DREG1D__) && !defined(__CUDA_SHMEM1DREG1D__)
	for (int k = 2 + k_offset; k < ns - 2; k += k_increment)
	{
#endif
#if defined(_OPENACC)
		#pragma acc loop independent
		for (int j = 2 + j_offset; j < ny - 2; j += j_increment)
		{
#else
		for (int j = j_offset; j < ny; j += j_increment)
		{
#if !defined(__CUDA_SHMEM2D__)
			if ((j < 2) || (j >= ny - 2)) continue;
#endif
#endif
#if defined(_OPENACC)
			#pragma acc loop independent vector(512)
#endif
			for (int i = i_offset; i < nx; i += i_increment)
			{

#if defined(__CUDACC__) && !defined(_PPCG) && ((defined(__CUDA_SHMEM2D__) && defined(__CUDA_SHMEM2DREG1D__)) || \
	(defined(__CUDA_SHMEM1D__) && defined(__CUDA_SHMEM1DREG1D__)))

				// Following Paulius Micikevicius: 2D slice in shared memory,
				// values of third dimension column are shared through small
				// array, that compiler maps on registers.
				int k = 2 + k_offset;
#if defined(__CUDA_VECTORIZE2__)
				real2 w1_reg[5];
#else
				real w1_reg[5];
#endif
				if (k < ns - 2)
				{
					_R1D(w1, 1) = _A(w1, k-2, j, i);
					_R1D(w1, 2) = _A(w1, k-1, j, i);
					_R1D(w1, 3) = _A(w1, k, j, i);
					_R1D(w1, 4) = _A(w1, k+1, j, i);
				}
				for ( ; k < ns - 2; k += k_increment)
				{
					_R1D(w1, 0) = _R1D(w1, 1);
					_R1D(w1, 1) = _R1D(w1, 2);
					_R1D(w1, 2) = _R1D(w1, 3);
					_R1D(w1, 3) = _R1D(w1, 4);
					_R1D(w1, 4) = _A(w1, k+2, j, i);

#endif

#if defined(__CUDACC__) && !defined(_PPCG) && defined(__CUDA_SHMEM1D__)

#if defined(__CUDA_VECTORIZE2__)
					int i_shm = 2 * threadIdx.x;
#else
					int i_shm = threadIdx.x;
#endif

					_S1D(w1, i_shm) = _A(w1, k, j, i);
					if (i_shm < 2)
					{
						_S1D(w1, i_shm - 2) = _A(w1, k, j, i - 2);
#if defined(__CUDA_VECTORIZE2__)
						if (i + 2 * blockDim.x < nx)
							_S1D(w1, i_shm + 2 * blockDim.x) = _A(w1, k, j, i + 2 * blockDim.x);
#else
						if (i + blockDim.x < nx)
							_S1D(w1, i_shm + blockDim.x) = _A(w1, k, j, i + blockDim.x);
#endif
					}
					__syncthreads();

					if ((i < 2) || (i >= nx - 2)) continue;

#if !defined(__CUDA_SHMEM1DREG1D__)

#if defined(__CUDA_VECTORIZE2__)

					real2* w2_k_j_i = &_A(w2, k, j, i);
					real2 w1_k_j_i = _S1D(w1, i_shm);
					real2 w0_k_j_i = _A(w0, k, j, i);

					real2 w1_k_j_ip2 = _S1D(w1, i_shm + 2);
					real2 w1_k_j_im2 = _S1D(w1, i_shm - 2);
					real2 w1_k_j_ip1; w1_k_j_ip1.x = w1_k_j_i.y; w1_k_j_ip1.y = w1_k_j_ip2.x;
					real2 w1_k_j_im1; w1_k_j_im1.x = w1_k_j_im2.y; w1_k_j_im1.y = w1_k_j_i.x;
					
					real2 w1_k_jp1_i = _A(w1, k, j+1, i);
					real2 w1_k_jm1_i = _A(w1, k, j-1, i);
					real2 w1_k_jp2_i = _A(w1, k, j+2, i);
					real2 w1_k_jm2_i = _A(w1, k, j-2, i);
					
					real2 w1_kp1_j_i = _A(w1, k+1, j, i);
					real2 w1_km1_j_i = _A(w1, k-1, j, i);
					real2 w1_kp2_j_i = _A(w1, k+2, j, i);
					real2 w1_km2_j_i = _A(w1, k-2, j, i);

					(*w2_k_j_i).x =  m0 * w1_k_j_i.x - w0_k_j_i.x +

						m1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						m2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  m0 * w1_k_j_i.y - w0_k_j_i.y +

						m1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						m2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);

#else

					_A(w2, k, j, i) =  m0 * _S1D(w1, i_shm) - _A(w0, k, j, i) +

						m1 * (
							_S1D(w1, i_shm + 1) + _S1D(w1, i_shm - 1) +
							_A(w1, k, j+1, i) + _A(w1, k, j-1, i)  +
							_A(w1, k+1, j, i) + _A(w1, k-1, j, i)) +
						m2 * (
							_S1D(w1, i_shm + 2) + _S1D(w1, i_shm - 2) +
							_A(w1, k, j+2, i) + _A(w1, k, j-2, i)  +
							_A(w1, k+2, j, i) + _A(w1, k-2, j, i));

#endif

#else

#if defined(__CUDA_VECTORIZE2__)

					real2* w2_k_j_i = &_A(w2, k, j, i);
					real2 w1_k_j_i = _S1D(w1, i_shm);
					real2 w0_k_j_i = _A(w0, k, j, i);

					real2 w1_k_j_ip2 = _S1D(w1, i_shm + 2);
					real2 w1_k_j_im2 = _S1D(w1, i_shm - 2);
					real2 w1_k_j_ip1; w1_k_j_ip1.x = w1_k_j_i.y; w1_k_j_ip1.y = w1_k_j_ip2.x;
					real2 w1_k_j_im1; w1_k_j_im1.x = w1_k_j_im2.y; w1_k_j_im1.y = w1_k_j_i.x;
					
					real2 w1_k_jp1_i = _A(w1, k, j+1, i);
					real2 w1_k_jm1_i = _A(w1, k, j-1, i);
					real2 w1_k_jp2_i = _A(w1, k, j+2, i);
					real2 w1_k_jm2_i = _A(w1, k, j-2, i);
					
					real2 w1_kp1_j_i = _R1D(w1, 3);
					real2 w1_km1_j_i = _R1D(w1, 1);
					real2 w1_kp2_j_i = _R1D(w1, 4);
					real2 w1_km2_j_i = _R1D(w1, 0);

					(*w2_k_j_i).x =  m0 * w1_k_j_i.x - w0_k_j_i.x +

						m1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						m2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  m0 * w1_k_j_i.y - w0_k_j_i.y +

						m1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						m2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);

#else

					_A(w2, k, j, i) =  m0 * _S1D(w1, i_shm) - _A(w0, k, j, i) +

						m1 * (
							_S1D(w1, i_shm + 1) + _S1D(w1, i_shm - 1) +
							_A(w1, k, j+1, i) + _A(w1, k, j-1, i)  +
							_R1D(w1, 3) + _R1D(w1, 1)) +
						m2 * (
							_S1D(w1, i_shm + 2) + _S1D(w1, i_shm - 2) +
							_A(w1, k, j+2, i) + _A(w1, k, j-2, i)  +
							_R1D(w1, 4) + _R1D(w1, 0));

#endif

#endif
					
#elif defined(__CUDACC__) && !defined(_PPCG) && defined(__CUDA_SHMEM2D__)

#if defined(__CUDA_VECTORIZE2__)
					int i_shm = 2 * threadIdx.x, j_shm = threadIdx.y;
#else
					int i_shm = threadIdx.x, j_shm = threadIdx.y;
#endif
					
					_S2D(w1, j_shm, i_shm) = _A(w1, k, j, i);
					if (j_shm < 2)
					{
						_S2D(w1, j_shm - 2, i_shm) = _A(w1, k, j - 2, i);
						if (j + blockDim.y < ny)
							_S2D(w1, j_shm + blockDim.y, i_shm) = _A(w1, k, j + blockDim.y, i);
					}
					if (i_shm < 2)
					{
						_S2D(w1, j_shm, i_shm - 2) = _A(w1, k, j, i - 2);
#if defined(__CUDA_VECTORIZE2__)
						if (i + 2 * blockDim.x < nx)
							_S2D(w1, j_shm, i_shm + 2 * blockDim.x) = _A(w1, k, j, i + 2 * blockDim.x);
#else
						if (i + blockDim.x < nx)
							_S2D(w1, j_shm, i_shm + blockDim.x) = _A(w1, k, j, i + blockDim.x);
#endif
					}
					__syncthreads();

					if ((j < 2) || (j >= ny - 2)) continue;
					if ((i < 2) || (i >= nx - 2)) continue;

#if !defined(__CUDA_SHMEM2DREG1D__)

#if defined(__CUDA_VECTORIZE2__)

					real2* w2_k_j_i = &_A(w2, k, j, i);
					real2 w1_k_j_i = _S2D(w1, j_shm, i_shm);
					real2 w0_k_j_i = _A(w0, k, j, i);

					real2 w1_k_j_ip2 = _S2D(w1, j_shm, i_shm + 2);
					real2 w1_k_j_im2 = _S2D(w1, j_shm, i_shm - 2);
					real2 w1_k_j_ip1; w1_k_j_ip1.x = w1_k_j_i.y; w1_k_j_ip1.y = w1_k_j_ip2.x;
					real2 w1_k_j_im1; w1_k_j_im1.x = w1_k_j_im2.y; w1_k_j_im1.y = w1_k_j_i.x;
					
					real2 w1_k_jp1_i = _S2D(w1, j_shm + 1, i_shm);
					real2 w1_k_jm1_i = _S2D(w1, j_shm - 1, i_shm);
					real2 w1_k_jp2_i = _S2D(w1, j_shm + 2, i_shm);
					real2 w1_k_jm2_i = _S2D(w1, j_shm - 2, i_shm);
					
					real2 w1_kp1_j_i = _A(w1, k+1, j, i);
					real2 w1_km1_j_i = _A(w1, k-1, j, i);
					real2 w1_kp2_j_i = _A(w1, k+2, j, i);
					real2 w1_km2_j_i = _A(w1, k-2, j, i);

					(*w2_k_j_i).x =  m0 * w1_k_j_i.x - w0_k_j_i.x +

						m1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						m2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  m0 * w1_k_j_i.y - w0_k_j_i.y +

						m1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						m2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);

#else

					_A(w2, k, j, i) =  m0 * _S2D(w1, j_shm, i_shm) - _A(w0, k, j, i) +

						m1 * (
							_S2D(w1, j_shm, i_shm + 1) + _S2D(w1, j_shm, i_shm - 1) +
							_S2D(w1, j_shm + 1, i_shm) + _S2D(w1, j_shm - 1, i_shm) +
							_A(w1, k+1, j, i) + _A(w1, k-1, j, i)) +
						m2 * (
							_S2D(w1, j_shm, i_shm + 2) + _S2D(w1, j_shm, i_shm - 2) +
							_S2D(w1, j_shm + 2, i_shm) + _S2D(w1, j_shm - 2, i_shm) +
							_A(w1, k+2, j, i) + _A(w1, k-2, j, i));

#endif

#else

#if defined(__CUDA_VECTORIZE2__)

					real2* w2_k_j_i = &_A(w2, k, j, i);
					real2 w1_k_j_i = _S2D(w1, j_shm, i_shm);
					real2 w0_k_j_i = _A(w0, k, j, i);

					real2 w1_k_j_ip2 = _S2D(w1, j_shm, i_shm + 2);
					real2 w1_k_j_im2 = _S2D(w1, j_shm, i_shm - 2);
					real2 w1_k_j_ip1; w1_k_j_ip1.x = w1_k_j_i.y; w1_k_j_ip1.y = w1_k_j_ip2.x;
					real2 w1_k_j_im1; w1_k_j_im1.x = w1_k_j_im2.y; w1_k_j_im1.y = w1_k_j_i.x;
					
					real2 w1_k_jp1_i = _S2D(w1, j_shm + 1, i_shm);
					real2 w1_k_jm1_i = _S2D(w1, j_shm - 1, i_shm);
					real2 w1_k_jp2_i = _S2D(w1, j_shm + 2, i_shm);
					real2 w1_k_jm2_i = _S2D(w1, j_shm - 2, i_shm);
					
					real2 w1_kp1_j_i = _R1D(w1, 3);
					real2 w1_km1_j_i = _R1D(w1, 1);
					real2 w1_kp2_j_i = _R1D(w1, 4);
					real2 w1_km2_j_i = _R1D(w1, 0);

					(*w2_k_j_i).x =  m0 * w1_k_j_i.x - w0_k_j_i.x +

						m1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						m2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  m0 * w1_k_j_i.y - w0_k_j_i.y +

						m1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						m2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);

#else

					_A(w2, k, j, i) =  m0 * _S2D(w1, j_shm, i_shm) - _A(w0, k, j, i) +

						m1 * (
							_S2D(w1, j_shm, i_shm + 1) + _S2D(w1, j_shm, i_shm - 1) +
							_S2D(w1, j_shm + 1, i_shm) + _S2D(w1, j_shm - 1, i_shm) +
							_R1D(w1, 3) + _R1D(w1, 1)) +
						m2 * (
							_S2D(w1, j_shm, i_shm + 2) + _S2D(w1, j_shm, i_shm - 2) +
							_S2D(w1, j_shm + 2, i_shm) + _S2D(w1, j_shm - 2, i_shm) +
							_R1D(w1, 4) + _R1D(w1, 0));

#endif

#endif

#else

#if !defined(__CUDA_SHUFFLE__)

					if ((i < 2) || (i >= nx - 2)) continue;

#endif

#if defined(__CUDA_VECTORIZE2__)

#if defined(__CUDA_SHUFFLE__)

					real2 val = _A(w1, k, j, i+2);
					real2 result; result.x = m2 * val.x; result.y = m2 * val.y;
					real swap = val.y;
					val.y = val.x;
					val.x = __shfl_up(swap, 1);
					if (laneid == 0)
						val.x = _AS(w1, k, j, i+1);
					result.x += m1 * val.x; result.y += m1 * val.y;
					swap = val.y;
					val.y = val.x;
					val.x = __shfl_up(swap, 1);
					if ((laneid == 0) || (laneid == 1))
						val.x = _AS(w1, k, j, i);
					result.x += m0 * val.x; result.y += m0 * val.y;
					swap = val.y;
					val.y = val.x;
					val.x = __shfl_up(swap, 1);
					if ((laneid == 0) || (laneid == 1) || (laneid == 2))
						val.x =_AS(w1, k, j, i-1);
					result.x += m1 * val.x; result.y += m1 * val.y;
					swap = val.y;
					val.y = val.x;
					val.x = __shfl_up(swap, 1);
					if ((laneid == 0) || (laneid == 1) || (laneid == 2) || (laneid == 3))
						val.x = _AS(w1, k, j, i-2);
					result.x += m2 * val.x; result.y += m2 * val.y;

					if ((i < 2) || (i >= nx - 2)) continue;

					real2* w2_k_j_i = &_A(w2, k, j, i);
					real2 w0_k_j_i = _A(w0, k, j, i);

					real2 w1_k_jp1_i = _A(w1, k, j+1, i);
					real2 w1_k_jm1_i = _A(w1, k, j-1, i);
					real2 w1_k_jp2_i = _A(w1, k, j+2, i);
					real2 w1_k_jm2_i = _A(w1, k, j-2, i);
					
					real2 w1_kp1_j_i = _A(w1, k+1, j, i);
					real2 w1_km1_j_i = _A(w1, k-1, j, i);
					real2 w1_kp2_j_i = _A(w1, k+2, j, i);
					real2 w1_km2_j_i = _A(w1, k-2, j, i);

					(*w2_k_j_i).x =  result.x - w0_k_j_i.x +

						m1 * (
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						m2 * (
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  result.y - w0_k_j_i.y +

						m1 * (
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						m2 * (
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);

#else

					real2* w2_k_j_i = &_A(w2, k, j, i);
					real2 w1_k_j_i = _A(w1, k, j, i);
					real2 w0_k_j_i = _A(w0, k, j, i);
					
					real2 w1_k_j_ip2 = _A(w1, k, j, i+2);
					real2 w1_k_j_im2 = _A(w1, k, j, i-2);
					real2 w1_k_j_ip1; w1_k_j_ip1.x = w1_k_j_i.y; w1_k_j_ip1.y = w1_k_j_ip2.x;
					real2 w1_k_j_im1; w1_k_j_im1.x = w1_k_j_im2.y; w1_k_j_im1.y = w1_k_j_i.x;
					
					real2 w1_k_jp1_i = _A(w1, k, j+1, i);
					real2 w1_k_jm1_i = _A(w1, k, j-1, i);
					real2 w1_k_jp2_i = _A(w1, k, j+2, i);
					real2 w1_k_jm2_i = _A(w1, k, j-2, i);
					
					real2 w1_kp1_j_i = _A(w1, k+1, j, i);
					real2 w1_km1_j_i = _A(w1, k-1, j, i);
					real2 w1_kp2_j_i = _A(w1, k+2, j, i);
					real2 w1_km2_j_i = _A(w1, k-2, j, i);

					(*w2_k_j_i).x =  m0 * w1_k_j_i.x - w0_k_j_i.x +

						m1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						m2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  m0 * w1_k_j_i.y - w0_k_j_i.y +

						m1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						m2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);

#endif

#elif defined(__CUDA_SHUFFLE__)

					real val = _A(w1, k, j, i+2);
					real result = m2 * val;
					val = __shfl_up(val, 1);
					if (laneid == 0)
						val = _A(w1, k, j, i+1);
					result += m1 * val;
					val = __shfl_up(val, 1);
					if ((laneid == 0) || (laneid == 1))
						val = _A(w1, k, j, i);
					result += m0 * val;
					val = __shfl_up(val, 1);
					if ((laneid == 0) || (laneid == 1) || (laneid == 2))
						val = _A(w1, k, j, i-1);
					result += m1 * val;
					val = __shfl_up(val, 1);
					if ((laneid == 0) || (laneid == 1) || (laneid == 2) || (laneid == 3))
						val = _A(w1, k, j, i-2);
					result += m2 * val;

					if ((i < 2) || (i >= nx - 2)) continue;

					_A(w2, k, j, i) = result - _A(w0, k, j, i) +

						m1 * (
							_A(w1, k, j+1, i) + _A(w1, k, j-1, i)  +
							_A(w1, k+1, j, i) + _A(w1, k-1, j, i)) +
						m2 * (
							_A(w1, k, j+2, i) + _A(w1, k, j-2, i)  +
							_A(w1, k+2, j, i) + _A(w1, k-2, j, i));

#else

					_A(w2, k, j, i) =  m0 * _A(w1, k, j, i) - _A(w0, k, j, i) +

						m1 * (
							_A(w1, k, j, i+1) + _A(w1, k, j, i-1)  +
							_A(w1, k, j+1, i) + _A(w1, k, j-1, i)  +
							_A(w1, k+1, j, i) + _A(w1, k-1, j, i)) +
						m2 * (
							_A(w1, k, j, i+2) + _A(w1, k, j, i-2)  +
							_A(w1, k, j+2, i) + _A(w1, k, j-2, i)  +
							_A(w1, k+2, j, i) + _A(w1, k-2, j, i));

#endif

#endif

#if defined(__CUDA_SHMEM2DREG1D__) || defined(__CUDA_SHMEM1DREG1D__)

				} // k-loop

#endif

			} // i-loop
		} // j-loop
		
#if !defined(__CUDA_SHMEM2DREG1D__) && !defined(__CUDA_SHMEM1DREG1D__)

	} // k-loop

#endif

#if defined(_PPCG)
	#pragma endscop
#endif

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

#if defined(__CUDA_VECTORIZE2__)
	if (nx % 2)
	{
		fprintf(stderr, "Vectorized version can't work with "
			"non-even X problem dimension (%d)\n", nx);
		exit(1);
	}
#endif
#if defined(__CUDA_VECTORIZE4__)
	fprintf(stderr, "4-element vectorization is not supported by this test\n");
	exit(1);
#endif

	real m0 = real_rand();
	real m1 = real_rand() / 6.;
	real m2 = real_rand() / 6.;

	printf("m0 = %f, m1 = %f, m2 = %f\n", m0, m1, m2);

	size_t szarray = (size_t)nx * ny * ns;
	size_t szarrayb = szarray * sizeof(real);

	real* w0 = (real*)memalign(MEMALIGN, szarrayb);
	real* w1 = (real*)memalign(MEMALIGN, szarrayb);
	real* w2 = (real*)memalign(MEMALIGN, szarrayb);

	if (!w0 || !w1 || !w2)
	{
		printf("Error allocating memory for arrays: %p, %p, %p\n", w0, w1, w2);
		exit(1);
	}

	real mean = 0.0f;
	for (int i = 0; i < szarray; i++)
	{
		w0[i] = real_rand();
		w1[i] = real_rand();
		w2[i] = real_rand();
		mean += w0[i] + w1[i] + w2[i];
	}
	printf("initial mean = %f\n", mean / szarray / 3);

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
		nocopy(w0:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(w1:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(w2:length(szarray) alloc_if(0) free_if(0))
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
#if defined(_MIC) || defined(_OPENACC) || (defined(__CUDACC__) && !defined(_PPCG))
	volatile struct timespec alloc_s, alloc_f;
#if defined(_MIC)
	get_time(&alloc_s);
	#pragma offload target(mic) \
		nocopy(w0:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(w1:length(szarray) alloc_if(1) free_if(0)), \
		nocopy(w2:length(szarray) alloc_if(1) free_if(0))
	{ }
	get_time(&alloc_f);
#endif
#if defined(_OPENACC)
	get_time(&alloc_s);
	#pragma acc data create (w0[0:szarray], w1[0:szarray], w2[0:szarray])
	{
	get_time(&alloc_f);
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
	get_time(&alloc_s);
	real *w0_dev = NULL, *w1_dev = NULL, *w2_dev = NULL;
	CUDA_SAFE_CALL(cudaMalloc(&w0_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&w1_dev, szarrayb));
	CUDA_SAFE_CALL(cudaMalloc(&w2_dev, szarrayb));
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
#if defined(_MIC) || defined(_OPENACC) || (defined(__CUDACC__) && !defined(_PPCG))
	volatile struct timespec load_s, load_f;
#if defined(_MIC)
	get_time(&load_s);
	#pragma offload target(mic) \
		in(w0:length(szarray) alloc_if(0) free_if(0)), \
		in(w1:length(szarray) alloc_if(0) free_if(0)), \
		in(w2:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&load_f);
#endif
#if defined(_OPENACC)
	get_time(&load_s);
	#pragma acc update device(w0[0:szarray], w1[0:szarray], w2[0:szarray])
	get_time(&load_f);
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
	get_time(&load_s);
	CUDA_SAFE_CALL(cudaMemcpy(w0_dev, w0, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(w1_dev, w1, szarrayb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(w2_dev, w2, szarrayb, cudaMemcpyHostToDevice));
	get_time(&load_f);
#endif
	double load_t = get_time_diff((struct timespec*)&load_s, (struct timespec*)&load_f);
	if (!no_timing) printf("data load time = %f sec (%f GB/sec)\n", load_t, 3 * szarrayb / (load_t * 1024 * 1024 * 1024));
#endif

	//
	// 4) Perform data processing iterations, keeping all data
	// on device.
	//
	int idxs[] = { 0, 1, 2 };
	volatile struct timespec compute_s, compute_f;
#if defined(__CUDACC__)
	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
#if !defined(_PPCG)
	kernelgen_cuda_config_t config;
#if (defined(__CUDA_SHMEM1D__) || defined(__CUDA_SHMEM2D__)) && \
	(defined(__CUDA_VECTORIZE2__) || defined(__CUDA_VECTORIZE4__))
	CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
#endif
	if (sizeof(real) == sizeof(double))
		CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
#if defined(__CUDA_SHMEM1D__)
	kernelgen_cuda_configure_gird(1, nx, ny, ns, &config);
	kernelgen_cuda_configure_shmem(&config, 1, sizeof(real),
		config.blockDim.x + STENCIL_BOUNDARY_LEFT + STENCIL_BOUNDARY_RIGHT,
		STENCIL_BOUNDARY_LEFT);
#elif defined(__CUDA_SHMEM2D__)
	kernelgen_cuda_configure_gird(2, nx, ny, ns, &config);
	kernelgen_cuda_configure_shmem(&config, 1, sizeof(real),
		(config.blockDim.x + STENCIL_BOUNDARY_LEFT + STENCIL_BOUNDARY_RIGHT + STENCIL_BANK_SHIFT) *
		(config.blockDim.y + STENCIL_BOUNDARY_TOP + STENCIL_BOUNDARY_BOTTOM), STENCIL_BOUNDARY_TOP *
		(config.blockDim.x + STENCIL_BOUNDARY_LEFT + STENCIL_BOUNDARY_RIGHT + STENCIL_BANK_SHIFT) +
		STENCIL_BOUNDARY_LEFT);
#else
	kernelgen_cuda_configure_gird(1, nx, ny, ns, &config);
#endif
#endif
#endif
	get_time(&compute_s);
#if defined(_MIC)
	#pragma offload target(mic) \
		nocopy(w0:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(w1:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(w2:length(szarray) alloc_if(0) free_if(0))
#endif
	{
#if !defined(__CUDACC__) || defined(_PPCG)
		real *w0p = w0, *w1p = w1, *w2p = w2;
#else
		real *w0p = w0_dev, *w1p = w1_dev, *w2p = w2_dev;
#endif
		for (int it = 0; it < nt; it++)
		{
#if !defined(__CUDACC__) || defined(_PPCG)
			wave13pt(nx, ny, ns, m0, m1, m2, w0p, w1p, w2p);
#else
			wave13pt<<<config.gridDim, config.blockDim, config.szshmem>>>(
				nx, ny, ns,
				config,
				m0, m1, m2, w0p, w1p, w2p);
			CUDA_SAFE_CALL(cudaGetLastError());
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
			real* w = w0p; w0p = w1p; w1p = w2p; w2p = w;
			int idx = idxs[0]; idxs[0] = idxs[1]; idxs[1] = idxs[2]; idxs[2] = idx;
		}
	}
	get_time(&compute_f);
	double compute_t = get_time_diff((struct timespec*)&compute_s, (struct timespec*)&compute_f);
	if (!no_timing) printf("compute time = %f sec\n", compute_t);

#if !defined(__CUDACC__) || defined(_PPCG)
	real* w[] = { w0, w1, w2 }; 
	w0 = w[idxs[0]]; w1 = w[idxs[1]]; w2 = w[idxs[2]];
#else
	real* w[] = { w0_dev, w1_dev, w2_dev }; 
	w0_dev = w[idxs[0]]; w1_dev = w[idxs[1]]; w2_dev = w[idxs[2]];
#if defined(__CUDA_SHMEM1D__) || defined(__CUDA_SHMEM2D__)
	kernelgen_cuda_config_dispose(&config);
#endif
#endif

	//
	// MIC or OPENACC or CUDA:
	//
	// 5) Transfer output data back from device to host.
	//
#if defined(_MIC) || defined(_OPENACC) || (defined(__CUDACC__) && !defined(_PPCG))
	volatile struct timespec save_s, save_f;
#if defined(_MIC)
	get_time(&save_s);
	#pragma offload target(mic) \
		out(w1:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&save_f);
#endif
#if defined(_OPENACC)
	get_time(&save_s);
	#pragma acc update host (w1[0:szarray])
	get_time(&save_f);
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
	get_time(&save_s);
	CUDA_SAFE_CALL(cudaMemcpy(w1, w1_dev, szarrayb, cudaMemcpyDeviceToHost));
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
#if defined(_MIC) || (defined(__CUDACC__) && !defined(_PPCG))
	volatile struct timespec free_s, free_f;
#if defined(_MIC)
	get_time(&free_s);
	#pragma offload target(mic) \
		nocopy(w0:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(w1:length(szarray) alloc_if(0) free_if(1)), \
		nocopy(w2:length(szarray) alloc_if(0) free_if(1))
	{ }
	get_time(&free_f);
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
	get_time(&free_s);
	CUDA_SAFE_CALL(cudaFree(w0_dev));
	CUDA_SAFE_CALL(cudaFree(w1_dev));
	CUDA_SAFE_CALL(cudaFree(w2_dev));
	get_time(&free_f);
#endif
	double free_t = get_time_diff((struct timespec*)&free_s, (struct timespec*)&free_f);
	if (!no_timing) printf("device buffer free time = %f sec\n", free_t);
#endif

	// For the final mean - account only the norm of the top
	// most level (tracked by swapping idxs array of indexes).
	mean = 0.0f;
	for (int i = 0; i < szarray; i++)
		mean += w1[i];
	printf("final mean = %f\n", mean / szarray);

	free(w0);
	free(w1);
	free(w2);

	fflush(stdout);

	return 0;
}

