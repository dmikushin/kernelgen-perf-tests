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

// Stencil boundaries thicknesses used to configure CUDA shared memory.
// In stencil code itself we still use plane numbers for better readability.
#if defined(__CUDA_SHMEM1D__) || defined(__CUDA_SHMEM2D__)
#define STENCIL_BOUNDARY_LEFT 1
#define STENCIL_BOUNDARY_RIGHT 2 
#define STENCIL_BOUNDARY_BOTTOM 1
#define STENCIL_BOUNDARY_TOP 2
#define STENCIL_BOUNDARY_BACK 1
#define STENCIL_BOUNDARY_FRONT 2
#define SHMEM_BANK_SHIFT 2
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

#if defined(_MIC)
__attribute__((target(mic)))
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
extern "C" __global__
#endif
void tricubic(int nx, int ny, int ns,
#if defined(__CUDACC__) && !defined(_PPCG)
	kernelgen_cuda_config_t config,
#endif
	const real* const __restrict__ u0p, real* const __restrict__ u1p,
	const real* const __restrict__ ap, const real* const __restrict__ bp,
	const real* const __restrict__ cp)
{
#if !defined(__CUDACC__) && defined(_PPCG)
	real (*u0)[ny][nx] = (real(*)[ny][nx])u0p;
	real (*u1)[ny][nx] = (real(*)[ny][nx])u1p;
	real (*a)[ny][nx] = (real(*)[ny][nx])ap;
	real (*b)[ny][nx] = (real(*)[ny][nx])bp;
	real (*c)[ny][nx] = (real(*)[ny][nx])cp;
#else
	const real* const __restrict__ u0 = u0p;
	real* const __restrict__ u1 = u1p;
	const real* const __restrict__ a = ap;
	const real* const __restrict__ b = bp;
	const real* const __restrict__ c = cp;
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
	tricubic_patus(&dummy, u0, u1, a, b, c, nx, ny, ns);
#else
#if defined(_OPENACC)
	size_t szarray = (size_t)nx * ny * ns;
	#pragma acc kernels loop independent gang(ns), present(u0[0:szarray], u1[0:szarray], a[0:szarray], b[0:szarray], c[0:szarray])
#endif
#if defined(_OPENMP) || defined(_MIC)
	#pragma omp parallel for
#endif
#if defined(_PPCG)
	#pragma scop
#endif
#if !defined(__CUDA_SHMEM2DREG1D__) && !defined(__CUDA_SHMEM1DREG1D__)
	for (int k = 1 + k_offset; k < ns - 2; k += k_increment)
	{
#endif
#if defined(_OPENACC)
		#pragma acc loop independent
		for (int j = 1 + j_offset; j < ny - 2; j += j_increment)
		{
#else
		for (int j = j_offset; j < ny; j += j_increment)
		{
#if !defined(__CUDA_SHMEM2D__)
			if ((j < 1) || (j >= ny - 2)) continue;
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
#if 0 // BEGIN TODO
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
#endif // END TODO
					if ((i < 1) || (i >= nx - 2)) continue;

#if !defined(__CUDA_SHMEM1DREG1D__)

#if defined(__CUDA_VECTORIZE2__)

#if 0 // BEGIN TODO
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

					(*w2_k_j_i).x =  c0 * w1_k_j_i.x - w0_k_j_i.x +

						c1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						c2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  c0 * w1_k_j_i.y - w0_k_j_i.y +

						c1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						c2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);
#endif // END TODO

#else

#if 0 // BEGIN TODO
					_A(w2, k, j, i) =  c0 * _S1D(w1, i_shm) - _A(w0, k, j, i) +

						c1 * (
							_S1D(w1, i_shm + 1) + _S1D(w1, i_shm - 1) +
							_A(w1, k, j+1, i) + _A(w1, k, j-1, i)  +
							_A(w1, k+1, j, i) + _A(w1, k-1, j, i)) +
						c2 * (
							_S1D(w1, i_shm + 2) + _S1D(w1, i_shm - 2) +
							_A(w1, k, j+2, i) + _A(w1, k, j-2, i)  +
							_A(w1, k+2, j, i) + _A(w1, k-2, j, i));
#endif // END TODO

#endif

#else

#if defined(__CUDA_VECTORIZE2__)

#if 0 // BEGIN TODO
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

					(*w2_k_j_i).x =  c0 * w1_k_j_i.x - w0_k_j_i.x +

						c1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						c2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  c0 * w1_k_j_i.y - w0_k_j_i.y +

						c1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						c2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);
#endif // END TODO

#else

#if 0 // BEGIN TODO
					_A(w2, k, j, i) =  c0 * _S1D(w1, i_shm) - _A(w0, k, j, i) +

						c1 * (
							_S1D(w1, i_shm + 1) + _S1D(w1, i_shm - 1) +
							_A(w1, k, j+1, i) + _A(w1, k, j-1, i)  +
							_R1D(w1, 3) + _R1D(w1, 1)) +
						c2 * (
							_S1D(w1, i_shm + 2) + _S1D(w1, i_shm - 2) +
							_A(w1, k, j+2, i) + _A(w1, k, j-2, i)  +
							_R1D(w1, 4) + _R1D(w1, 0));
#endif // END TODO

#endif

#endif
					
#elif defined(__CUDACC__) && !defined(_PPCG) && defined(__CUDA_SHMEM2D__)

#if defined(__CUDA_VECTORIZE2__)
					int i_shm = 2 * threadIdx.x, j_shm = threadIdx.y;
#else
					int i_shm = threadIdx.x, j_shm = threadIdx.y;
#endif
					
					_S2D(w1, j_shm, i_shm) = _A(w1, k, j, i);
#if 0 // BEGIN TODO
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
#endif // END TODO
					if ((j < 1) || (j >= ny - 2)) continue;
					if ((i < 1) || (i >= nx - 2)) continue;

#if !defined(__CUDA_SHMEM2DREG1D__)

#if defined(__CUDA_VECTORIZE2__)

#if 0 // BEGIN TODO
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

					(*w2_k_j_i).x =  c0 * w1_k_j_i.x - w0_k_j_i.x +

						c1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						c2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  c0 * w1_k_j_i.y - w0_k_j_i.y +

						c1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						c2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);
#endif // END TODO

#else

#if 0 // BEGIN TODO
					_A(w2, k, j, i) =  c0 * _S2D(w1, j_shm, i_shm) - _A(w0, k, j, i) +

						c1 * (
							_S2D(w1, j_shm, i_shm + 1) + _S2D(w1, j_shm, i_shm - 1) +
							_S2D(w1, j_shm + 1, i_shm) + _S2D(w1, j_shm - 1, i_shm) +
							_A(w1, k+1, j, i) + _A(w1, k-1, j, i)) +
						c2 * (
							_S2D(w1, j_shm, i_shm + 2) + _S2D(w1, j_shm, i_shm - 2) +
							_S2D(w1, j_shm + 2, i_shm) + _S2D(w1, j_shm - 2, i_shm) +
							_A(w1, k+2, j, i) + _A(w1, k-2, j, i));
#endif // END TODO

#endif

#else

#if defined(__CUDA_VECTORIZE2__)

#if 0 // BEGIN TODO
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

					(*w2_k_j_i).x =  c0 * w1_k_j_i.x - w0_k_j_i.x +

						c1 * (
							w1_k_j_ip1.x + w1_k_j_im1.x +
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						c2 * (
							w1_k_j_ip2.x + w1_k_j_im2.x +
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  c0 * w1_k_j_i.y - w0_k_j_i.y +

						c1 * (
							w1_k_j_ip1.y + w1_k_j_im1.y +
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						c2 * (
							w1_k_j_ip2.y + w1_k_j_im2.y +
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);
#endif // END TODO

#else

#if 0 // BEGIN TODO
					_A(w2, k, j, i) =  c0 * _S2D(w1, j_shm, i_shm) - _A(w0, k, j, i) +

						c1 * (
							_S2D(w1, j_shm, i_shm + 1) + _S2D(w1, j_shm, i_shm - 1) +
							_S2D(w1, j_shm + 1, i_shm) + _S2D(w1, j_shm - 1, i_shm) +
							_R1D(w1, 3) + _R1D(w1, 1)) +
						c2 * (
							_S2D(w1, j_shm, i_shm + 2) + _S2D(w1, j_shm, i_shm - 2) +
							_S2D(w1, j_shm + 2, i_shm) + _S2D(w1, j_shm - 2, i_shm) +
							_R1D(w1, 4) + _R1D(w1, 0));
#endif // END TODO

#endif

#endif

#else

#if !defined(__CUDA_SHUFFLE__)

#if !defined(__CUDA_VECTORIZE2__)
					if ((i < 1) || (i >= nx - 2)) continue;
#else
					if (i >= nx - 2) continue;
#endif

#endif

#if defined(__CUDA_VECTORIZE2__)

#if defined(__CUDA_SHUFFLE__)

#if 0 // BEGIN TODO
					real2 val = _A(w1, k, j, i+2);
					real2 result; result.x = c2 * val.x; result.y = c2 * val.y;
					real swap = val.y;
					val.y = val.x;
					val.x = __shfl_up(swap, 1);
					if (laneid == 0)
						val.x = _AS(w1, k, j, i+1);
					result.x += c1 * val.x; result.y += c1 * val.y;
					swap = val.y;
					val.y = val.x;
					val.x = __shfl_up(swap, 1);
					if ((laneid == 0) || (laneid == 1))
						val.x = _AS(w1, k, j, i);
					result.x += c0 * val.x; result.y += c0 * val.y;
					swap = val.y;
					val.y = val.x;
					val.x = __shfl_up(swap, 1);
					if ((laneid == 0) || (laneid == 1) || (laneid == 2))
						val.x =_AS(w1, k, j, i-1);
					result.x += c1 * val.x; result.y += c1 * val.y;
					swap = val.y;
					val.y = val.x;
					val.x = __shfl_up(swap, 1);
					if ((laneid == 0) || (laneid == 1) || (laneid == 2) || (laneid == 3))
						val.x = _AS(w1, k, j, i-2);
					result.x += c2 * val.x; result.y += c2 * val.y;

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

						c1 * (
							w1_k_jp1_i.x + w1_k_jm1_i.x +
							w1_kp1_j_i.x + w1_km1_j_i.x) +
						c2 * (
							w1_k_jp2_i.x + w1_k_jm2_i.x +
							w1_kp2_j_i.x + w1_km2_j_i.x);

					(*w2_k_j_i).y =  result.y - w0_k_j_i.y +

						c1 * (
							w1_k_jp1_i.y + w1_k_jm1_i.y +
							w1_kp1_j_i.y + w1_km1_j_i.y) +
						c2 * (
							w1_k_jp2_i.y + w1_k_jm2_i.y +
							w1_kp2_j_i.y + w1_km2_j_i.y);
#endif // END TODO

#else

					real2 a_k_j_i = _A(a, k, j, i);
					real2 b_k_j_i = _A(b, k, j, i);
					real2 c_k_j_i = _A(c, k, j, i);

					real2 w1_a;
					real2 w2_a;
					real2 w3_a;
					real2 w4_a;
					real2 w1_b;
					real2 w2_b;
					real2 w3_b;
					real2 w4_b;
					real2 w1_c;
					real2 w2_c;
					real2 w3_c;
					real2 w4_c;

					// This branch should be eliminated in compile-time.
					if (sizeof(real) == sizeof(float))
					{
						w1_a.x =  1.0f/6.0f                      * a_k_j_i.x * (a_k_j_i.x + 1.0f) * (a_k_j_i.x + 2.0f);
						w1_a.y =  1.0f/6.0f                      * a_k_j_i.y * (a_k_j_i.y + 1.0f) * (a_k_j_i.y + 2.0f);
						w2_a.x = -0.5f      * (a_k_j_i.x - 1.0f)             * (a_k_j_i.x + 1.0f) * (a_k_j_i.x + 2.0f);
						w2_a.y = -0.5f      * (a_k_j_i.y - 1.0f)             * (a_k_j_i.y + 1.0f) * (a_k_j_i.y + 2.0f);
						w3_a.x =  0.5f      * (a_k_j_i.x - 1.0f) * a_k_j_i.x                      * (a_k_j_i.x + 2.0f);
						w3_a.y =  0.5f      * (a_k_j_i.y - 1.0f) * a_k_j_i.y                      * (a_k_j_i.y + 2.0f);
						w4_a.x = -1.0f/6.0f * (a_k_j_i.x - 1.0f) * a_k_j_i.x * (a_k_j_i.x + 1.0f);
						w4_a.y = -1.0f/6.0f * (a_k_j_i.y - 1.0f) * a_k_j_i.y * (a_k_j_i.y + 1.0f);
						w1_b.x =  1.0f/6.0f                      * b_k_j_i.x * (b_k_j_i.x + 1.0f) * (b_k_j_i.x + 2.0f);
						w1_b.y =  1.0f/6.0f                      * b_k_j_i.y * (b_k_j_i.y + 1.0f) * (b_k_j_i.y + 2.0f);
						w2_b.x = -0.5f      * (b_k_j_i.x - 1.0f)             * (b_k_j_i.x + 1.0f) * (b_k_j_i.x + 2.0f);
						w2_b.y = -0.5f      * (b_k_j_i.y - 1.0f)             * (b_k_j_i.y + 1.0f) * (b_k_j_i.y + 2.0f);
						w3_b.x =  0.5f      * (b_k_j_i.x - 1.0f) * b_k_j_i.x                      * (b_k_j_i.x + 2.0f);
						w3_b.y =  0.5f      * (b_k_j_i.y - 1.0f) * b_k_j_i.y                      * (b_k_j_i.y + 2.0f);
						w4_b.x = -1.0f/6.0f * (b_k_j_i.x - 1.0f) * b_k_j_i.x * (b_k_j_i.x + 1.0f);
						w4_b.y = -1.0f/6.0f * (b_k_j_i.y - 1.0f) * b_k_j_i.y * (b_k_j_i.y + 1.0f);
						w1_c.x =  1.0f/6.0f                      * c_k_j_i.x * (c_k_j_i.x + 1.0f) * (c_k_j_i.x + 2.0f);
						w1_c.y =  1.0f/6.0f                      * c_k_j_i.y * (c_k_j_i.y + 1.0f) * (c_k_j_i.y + 2.0f);
						w2_c.x = -0.5f      * (c_k_j_i.x - 1.0f)             * (c_k_j_i.x + 1.0f) * (c_k_j_i.x + 2.0f);
						w2_c.y = -0.5f      * (c_k_j_i.y - 1.0f)             * (c_k_j_i.y + 1.0f) * (c_k_j_i.y + 2.0f);
						w3_c.x =  0.5f      * (c_k_j_i.x - 1.0f) * c_k_j_i.x                      * (c_k_j_i.x + 2.0f);
						w3_c.y =  0.5f      * (c_k_j_i.y - 1.0f) * c_k_j_i.y                      * (c_k_j_i.y + 2.0f);
						w4_c.x = -1.0f/6.0f * (c_k_j_i.x - 1.0f) * c_k_j_i.x * (c_k_j_i.x + 1.0f);
						w4_c.y = -1.0f/6.0f * (c_k_j_i.y - 1.0f) * c_k_j_i.y * (c_k_j_i.y + 1.0f);
					}
					else
					{
						w1_a.x =  1.0 /6.0                       * a_k_j_i.x * (a_k_j_i.x + 1.0 ) * (a_k_j_i.x + 2.0 );
						w1_a.y =  1.0 /6.0                       * a_k_j_i.y * (a_k_j_i.y + 1.0 ) * (a_k_j_i.y + 2.0 );
						w2_a.x = -0.5       * (a_k_j_i.x - 1.0 )             * (a_k_j_i.x + 1.0 ) * (a_k_j_i.x + 2.0 );
						w2_a.y = -0.5       * (a_k_j_i.y - 1.0 )             * (a_k_j_i.y + 1.0 ) * (a_k_j_i.y + 2.0 );
						w3_a.x =  0.5       * (a_k_j_i.x - 1.0 ) * a_k_j_i.x                      * (a_k_j_i.x + 2.0 );
						w3_a.y =  0.5       * (a_k_j_i.y - 1.0 ) * a_k_j_i.y                      * (a_k_j_i.y + 2.0 );
						w4_a.x = -1.0 /6.0  * (a_k_j_i.x - 1.0 ) * a_k_j_i.x * (a_k_j_i.x + 1.0 );
						w4_a.y = -1.0 /6.0  * (a_k_j_i.y - 1.0 ) * a_k_j_i.y * (a_k_j_i.y + 1.0 );
						w1_b.x =  1.0 /6.0                       * b_k_j_i.x * (b_k_j_i.x + 1.0 ) * (b_k_j_i.x + 2.0 );
						w1_b.y =  1.0 /6.0                       * b_k_j_i.y * (b_k_j_i.y + 1.0 ) * (b_k_j_i.y + 2.0 );
						w2_b.x = -0.5       * (b_k_j_i.x - 1.0 )             * (b_k_j_i.x + 1.0 ) * (b_k_j_i.x + 2.0 );
						w2_b.y = -0.5       * (b_k_j_i.y - 1.0 )             * (b_k_j_i.y + 1.0 ) * (b_k_j_i.y + 2.0 );
						w3_b.x =  0.5       * (b_k_j_i.x - 1.0 ) * b_k_j_i.x                      * (b_k_j_i.x + 2.0 );
						w3_b.y =  0.5       * (b_k_j_i.y - 1.0 ) * b_k_j_i.y                      * (b_k_j_i.y + 2.0 );
						w4_b.x = -1.0 /6.0  * (b_k_j_i.x - 1.0 ) * b_k_j_i.x * (b_k_j_i.x + 1.0 );
						w4_b.y = -1.0 /6.0  * (b_k_j_i.y - 1.0 ) * b_k_j_i.y * (b_k_j_i.y + 1.0 );
						w1_c.x =  1.0 /6.0                       * c_k_j_i.x * (c_k_j_i.x + 1.0 ) * (c_k_j_i.x + 2.0 );
						w1_c.y =  1.0 /6.0                       * c_k_j_i.y * (c_k_j_i.y + 1.0 ) * (c_k_j_i.y + 2.0 );
						w2_c.x = -0.5       * (c_k_j_i.x - 1.0 )             * (c_k_j_i.x + 1.0 ) * (c_k_j_i.x + 2.0 );
						w2_c.y = -0.5       * (c_k_j_i.y - 1.0 )             * (c_k_j_i.y + 1.0 ) * (c_k_j_i.y + 2.0 );
						w3_c.x =  0.5       * (c_k_j_i.x - 1.0 ) * c_k_j_i.x                      * (c_k_j_i.x + 2.0 );
						w3_c.y =  0.5       * (c_k_j_i.y - 1.0 ) * c_k_j_i.y                      * (c_k_j_i.y + 2.0 );
						w4_c.x = -1.0 /6.0  * (c_k_j_i.x - 1.0 ) * c_k_j_i.x * (c_k_j_i.x + 1.0 );
						w4_c.y = -1.0 /6.0  * (c_k_j_i.y - 1.0 ) * c_k_j_i.y * (c_k_j_i.y + 1.0 );
					}

					real u1_x = 0, u1_y = 0;

					{
						real2 u0_km1_jm1_im2;
						u0_km1_jm1_im2 = _A(u0, k-1, j-1, (i-2) * !!i);
						real2 u0_km1_jm1_i = _A(u0, k-1, j-1, i);
						real2 u0_km1_jm1_im1; u0_km1_jm1_im1.x = u0_km1_jm1_im2.y; u0_km1_jm1_im1.y = u0_km1_jm1_i.x;
						real2 u0_km1_jm1_ip2 = _A(u0, k-1, j-1, i+2);
						real2 u0_km1_jm1_ip1; u0_km1_jm1_ip1.x = u0_km1_jm1_i.y; u0_km1_jm1_ip1.y = u0_km1_jm1_ip2.x;

						u1_x += w1_a.x * w1_b.x * w1_c.x * u0_km1_jm1_im1.x +
							w2_a.x * w1_b.x * w1_c.x * u0_km1_jm1_i.x   +
							w3_a.x * w1_b.x * w1_c.x * u0_km1_jm1_ip1.x +
							w4_a.x * w1_b.x * w1_c.x * u0_km1_jm1_ip2.x;
							
						u1_y += w1_a.y * w1_b.y * w1_c.y * u0_km1_jm1_im1.y +
							w2_a.y * w1_b.y * w1_c.y * u0_km1_jm1_i.y   +
							w3_a.y * w1_b.y * w1_c.y * u0_km1_jm1_ip1.y +
							w4_a.y * w1_b.y * w1_c.y * u0_km1_jm1_ip2.y;
					}

					{
						real2 u0_km1_j_im2;
						u0_km1_j_im2 = _A(u0, k-1, j, (i-2) * !!i);
						real2 u0_km1_j_i = _A(u0, k-1, j, i);
						real2 u0_km1_j_im1; u0_km1_j_im1.x = u0_km1_j_im2.y; u0_km1_j_im1.y = u0_km1_j_i.x;
						real2 u0_km1_j_ip2 = _A(u0, k-1, j, i+2);
						real2 u0_km1_j_ip1; u0_km1_j_ip1.x = u0_km1_j_i.y; u0_km1_j_ip1.y = u0_km1_j_ip2.x;

						u1_x += w1_a.x * w2_b.x * w1_c.x * u0_km1_j_im1.x +
							w2_a.x * w2_b.x * w1_c.x * u0_km1_j_i.x   +
							w3_a.x * w2_b.x * w1_c.x * u0_km1_j_ip1.x +
							w4_a.x * w2_b.x * w1_c.x * u0_km1_j_ip2.x;

						u1_y += w1_a.y * w2_b.y * w1_c.y * u0_km1_j_im1.y +
							w2_a.y * w2_b.y * w1_c.y * u0_km1_j_i.y   +
							w3_a.y * w2_b.y * w1_c.y * u0_km1_j_ip1.y +
							w4_a.y * w2_b.y * w1_c.y * u0_km1_j_ip2.y;
					}

					{
						real2 u0_km1_jp1_im2;
						u0_km1_jp1_im2 = _A(u0, k-1, j+1, (i-2) * !!i);
						real2 u0_km1_jp1_i = _A(u0, k-1, j+1, i);
						real2 u0_km1_jp1_im1; u0_km1_jp1_im1.x = u0_km1_jp1_im2.y; u0_km1_jp1_im1.y = u0_km1_jp1_i.x;
						real2 u0_km1_jp1_ip2 = _A(u0, k-1, j+1, i+2);
						real2 u0_km1_jp1_ip1; u0_km1_jp1_ip1.x = u0_km1_jp1_i.y; u0_km1_jp1_ip1.y = u0_km1_jp1_ip2.x;

						u1_x += w1_a.x * w3_b.x * w1_c.x * u0_km1_jp1_im1.x +
							w2_a.x * w3_b.x * w1_c.x * u0_km1_jp1_i.x   +
							w3_a.x * w3_b.x * w1_c.x * u0_km1_jp1_ip1.x +
							w4_a.x * w3_b.x * w1_c.x * u0_km1_jp1_ip2.x;

						u1_y += w1_a.y * w3_b.y * w1_c.y * u0_km1_jp1_im1.y +
							w2_a.y * w3_b.y * w1_c.y * u0_km1_jp1_i.y   +
							w3_a.y * w3_b.y * w1_c.y * u0_km1_jp1_ip1.y +
							w4_a.y * w3_b.y * w1_c.y * u0_km1_jp1_ip2.y;
					}

					{
						real2 u0_km1_jp2_im2;
						u0_km1_jp2_im2 = _A(u0, k-1, j+2, (i-2) * !!i);
						real2 u0_km1_jp2_i = _A(u0, k-1, j+2, i);
						real2 u0_km1_jp2_im1; u0_km1_jp2_im1.x = u0_km1_jp2_im2.y; u0_km1_jp2_im1.y = u0_km1_jp2_i.x;
						real2 u0_km1_jp2_ip2 = _A(u0, k-1, j+2, i+2);
						real2 u0_km1_jp2_ip1; u0_km1_jp2_ip1.x = u0_km1_jp2_i.y; u0_km1_jp2_ip1.y = u0_km1_jp2_ip2.x;		

						u1_x += w1_a.x * w4_b.x * w1_c.x * u0_km1_jp2_im1.x +
							w2_a.x * w4_b.x * w1_c.x * u0_km1_jp2_i.x   +
							w3_a.x * w4_b.x * w1_c.x * u0_km1_jp2_ip1.x +
							w4_a.x * w4_b.x * w1_c.x * u0_km1_jp2_ip2.x;

						u1_y += w1_a.y * w4_b.y * w1_c.y * u0_km1_jp2_im1.y +
							w2_a.y * w4_b.y * w1_c.y * u0_km1_jp2_i.y   +
							w3_a.y * w4_b.y * w1_c.y * u0_km1_jp2_ip1.y +
							w4_a.y * w4_b.y * w1_c.y * u0_km1_jp2_ip2.y;
					}


					{
						real2 u0_k_jm1_im2;
						u0_k_jm1_im2 = _A(u0, k, j-1, (i-2) * !!i);
						real2 u0_k_jm1_i = _A(u0, k, j-1, i);
						real2 u0_k_jm1_im1; u0_k_jm1_im1.x = u0_k_jm1_im2.y; u0_k_jm1_im1.y = u0_k_jm1_i.x;
						real2 u0_k_jm1_ip2 = _A(u0, k, j-1, i+2);
						real2 u0_k_jm1_ip1; u0_k_jm1_ip1.x = u0_k_jm1_i.y; u0_k_jm1_ip1.y = u0_k_jm1_ip2.x;

						u1_x += w1_a.x * w1_b.x * w2_c.x * u0_k_jm1_im1.x +
							w2_a.x * w1_b.x * w2_c.x * u0_k_jm1_i.x   +
							w3_a.x * w1_b.x * w2_c.x * u0_k_jm1_ip1.x +
							w4_a.x * w1_b.x * w2_c.x * u0_k_jm1_ip2.x;

						u1_y += w1_a.y * w1_b.y * w2_c.y * u0_k_jm1_im1.y +
							w2_a.y * w1_b.y * w2_c.y * u0_k_jm1_i.y   +
							w3_a.y * w1_b.y * w2_c.y * u0_k_jm1_ip1.y +
							w4_a.y * w1_b.y * w2_c.y * u0_k_jm1_ip2.y;
					}			

					{
						real2 u0_k_j_im2;
						u0_k_j_im2 = _A(u0, k, j, (i-2) * !!i);
						real2 u0_k_j_i = _A(u0, k, j, i);
						real2 u0_k_j_im1; u0_k_j_im1.x = u0_k_j_im2.y; u0_k_j_im1.y = u0_k_j_i.x;
						real2 u0_k_j_ip2 = _A(u0, k, j, i+2);
						real2 u0_k_j_ip1; u0_k_j_ip1.x = u0_k_j_i.y; u0_k_j_ip1.y = u0_k_j_ip2.x;

						u1_x += w1_a.x * w2_b.x * w2_c.x * u0_k_j_im1.x +
							w2_a.x * w2_b.x * w2_c.x * u0_k_j_i.x   +
							w3_a.x * w2_b.x * w2_c.x * u0_k_j_ip1.x +
							w4_a.x * w2_b.x * w2_c.x * u0_k_j_ip2.x;

						u1_y += w1_a.y * w2_b.y * w2_c.y * u0_k_j_im1.y +
							w2_a.y * w2_b.y * w2_c.y * u0_k_j_i.y   +
							w3_a.y * w2_b.y * w2_c.y * u0_k_j_ip1.y +
							w4_a.y * w2_b.y * w2_c.y * u0_k_j_ip2.y;
					}

					{
						real2 u0_k_jp1_im2;
						u0_k_jp1_im2 = _A(u0, k, j+1, (i-2) * !!i);
						real2 u0_k_jp1_i = _A(u0, k, j+1, i);
						real2 u0_k_jp1_im1; u0_k_jp1_im1.x = u0_k_jp1_im2.y; u0_k_jp1_im1.y = u0_k_jp1_i.x;
						real2 u0_k_jp1_ip2 = _A(u0, k, j+1, i+2);
						real2 u0_k_jp1_ip1; u0_k_jp1_ip1.x = u0_k_jp1_i.y; u0_k_jp1_ip1.y = u0_k_jp1_ip2.x;

						u1_x += w1_a.x * w3_b.x * w2_c.x * u0_k_jp1_im1.x +
							w2_a.x * w3_b.x * w2_c.x * u0_k_jp1_i.x   +
							w3_a.x * w3_b.x * w2_c.x * u0_k_jp1_ip1.x +
							w4_a.x * w3_b.x * w2_c.x * u0_k_jp1_ip2.x;

						u1_y += w1_a.y * w3_b.y * w2_c.y * u0_k_jp1_im1.y +
							w2_a.y * w3_b.y * w2_c.y * u0_k_jp1_i.y   +
							w3_a.y * w3_b.y * w2_c.y * u0_k_jp1_ip1.y +
							w4_a.y * w3_b.y * w2_c.y * u0_k_jp1_ip2.y;
					}
					
					{
						real2 u0_k_jp2_im2;
						u0_k_jp2_im2 = _A(u0, k, j+2, (i-2) * !!i);
						real2 u0_k_jp2_i = _A(u0, k, j+2, i);
						real2 u0_k_jp2_im1; u0_k_jp2_im1.x = u0_k_jp2_im2.y; u0_k_jp2_im1.y = u0_k_jp2_i.x;
						real2 u0_k_jp2_ip2 = _A(u0, k, j+2, i+2);
						real2 u0_k_jp2_ip1; u0_k_jp2_ip1.x = u0_k_jp2_i.y; u0_k_jp2_ip1.y = u0_k_jp2_ip2.x;		

						u1_x += w1_a.x * w4_b.x * w2_c.x * u0_k_jp2_im1.x +
							w2_a.x * w4_b.x * w2_c.x * u0_k_jp2_i.x   +
							w3_a.x * w4_b.x * w2_c.x * u0_k_jp2_ip1.x +
							w4_a.x * w4_b.x * w2_c.x * u0_k_jp2_ip2.x;

						u1_y += w1_a.y * w4_b.y * w2_c.y * u0_k_jp2_im1.y +
							w2_a.y * w4_b.y * w2_c.y * u0_k_jp2_i.y   +
							w3_a.y * w4_b.y * w2_c.y * u0_k_jp2_ip1.y +
							w4_a.y * w4_b.y * w2_c.y * u0_k_jp2_ip2.y;
					}


					{
						real2 u0_kp1_jm1_im2;
						u0_kp1_jm1_im2 = _A(u0, k+1, j-1, (i-2) * !!i);
						real2 u0_kp1_jm1_i = _A(u0, k+1, j-1, i);
						real2 u0_kp1_jm1_im1; u0_kp1_jm1_im1.x = u0_kp1_jm1_im2.y; u0_kp1_jm1_im1.y = u0_kp1_jm1_i.x;
						real2 u0_kp1_jm1_ip2 = _A(u0, k+1, j-1, i+2);
						real2 u0_kp1_jm1_ip1; u0_kp1_jm1_ip1.x = u0_kp1_jm1_i.y; u0_kp1_jm1_ip1.y = u0_kp1_jm1_ip2.x;

						u1_x += w1_a.x * w1_b.x * w3_c.x * u0_kp1_jm1_im1.x +
							w2_a.x * w1_b.x * w3_c.x * u0_kp1_jm1_i.x +
							w3_a.x * w1_b.x * w3_c.x * u0_kp1_jm1_ip1.x +
							w4_a.x * w1_b.x * w3_c.x * u0_kp1_jm1_ip2.x;

						u1_y += w1_a.y * w1_b.y * w3_c.y * u0_kp1_jm1_im1.y +
							w2_a.y * w1_b.y * w3_c.y * u0_kp1_jm1_i.y +
							w3_a.y * w1_b.y * w3_c.y * u0_kp1_jm1_ip1.y +
							w4_a.y * w1_b.y * w3_c.y * u0_kp1_jm1_ip2.y;
					}			

					{
						real2 u0_kp1_j_im2;
						u0_kp1_j_im2 = _A(u0, k+1, j, (i-2) * !!i);
						real2 u0_kp1_j_i = _A(u0, k+1, j, i);
						real2 u0_kp1_j_im1; u0_kp1_j_im1.x = u0_kp1_j_im2.y; u0_kp1_j_im1.y = u0_kp1_j_i.x;
						real2 u0_kp1_j_ip2 = _A(u0, k+1, j, i+2);
						real2 u0_kp1_j_ip1; u0_kp1_j_ip1.x = u0_kp1_j_i.y; u0_kp1_j_ip1.y = u0_kp1_j_ip2.x;

						u1_x += w1_a.x * w2_b.x * w3_c.x * u0_kp1_j_im1.x +
							w2_a.x * w2_b.x * w3_c.x * u0_kp1_j_i.x +
							w3_a.x * w2_b.x * w3_c.x * u0_kp1_j_ip1.x +
							w4_a.x * w2_b.x * w3_c.x * u0_kp1_j_ip2.x;

						u1_y += w1_a.y * w2_b.y * w3_c.y * u0_kp1_j_im1.y +
							w2_a.y * w2_b.y * w3_c.y * u0_kp1_j_i.y +
							w3_a.y * w2_b.y * w3_c.y * u0_kp1_j_ip1.y +
							w4_a.y * w2_b.y * w3_c.y * u0_kp1_j_ip2.y;
					}

					{
						real2 u0_kp1_jp1_im2;
						u0_kp1_jp1_im2 = _A(u0, k+1, j+1, (i-2) * !!i);
						real2 u0_kp1_jp1_i = _A(u0, k+1, j+1, i);
						real2 u0_kp1_jp1_im1; u0_kp1_jp1_im1.x = u0_kp1_jp1_im2.y; u0_kp1_jp1_im1.y = u0_kp1_jp1_i.x;
						real2 u0_kp1_jp1_ip2 = _A(u0, k+1, j+1, i+2);
						real2 u0_kp1_jp1_ip1; u0_kp1_jp1_ip1.x = u0_kp1_jp1_i.y; u0_kp1_jp1_ip1.y = u0_kp1_jp1_ip2.x;

						u1_x += w1_a.x * w3_b.x * w3_c.x * u0_kp1_jp1_im1.x +
							w2_a.x * w3_b.x * w3_c.x * u0_kp1_jp1_i.x +
							w3_a.x * w3_b.x * w3_c.x * u0_kp1_jp1_ip1.x +
							w4_a.x * w3_b.x * w3_c.x * u0_kp1_jp1_ip2.x;

						u1_y += w1_a.y * w3_b.y * w3_c.y * u0_kp1_jp1_im1.y +
							w2_a.y * w3_b.y * w3_c.y * u0_kp1_jp1_i.y +
							w3_a.y * w3_b.y * w3_c.y * u0_kp1_jp1_ip1.y +
							w4_a.y * w3_b.y * w3_c.y * u0_kp1_jp1_ip2.y;
					}
					
					{
						real2 u0_kp1_jp2_im2;
						u0_kp1_jp2_im2 = _A(u0, k+1, j+2, (i-2) * !!i);
						real2 u0_kp1_jp2_i = _A(u0, k+1, j+2, i);
						real2 u0_kp1_jp2_im1; u0_kp1_jp2_im1.x = u0_kp1_jp2_im2.y; u0_kp1_jp2_im1.y = u0_kp1_jp2_i.x;
						real2 u0_kp1_jp2_ip2 = _A(u0, k+1, j+2, i+2);
						real2 u0_kp1_jp2_ip1; u0_kp1_jp2_ip1.x = u0_kp1_jp2_i.y; u0_kp1_jp2_ip1.y = u0_kp1_jp2_ip2.x;		

						u1_x += w1_a.x * w4_b.x * w3_c.x * u0_kp1_jp2_im1.x +
							w2_a.x * w4_b.x * w3_c.x * u0_kp1_jp2_i.x +
							w3_a.x * w4_b.x * w3_c.x * u0_kp1_jp2_ip1.x +
							w4_a.x * w4_b.x * w3_c.x * u0_kp1_jp2_ip2.x;

						u1_y += w1_a.y * w4_b.y * w3_c.y * u0_kp1_jp2_im1.y +
							w2_a.y * w4_b.y * w3_c.y * u0_kp1_jp2_i.y +
							w3_a.y * w4_b.y * w3_c.y * u0_kp1_jp2_ip1.y +
							w4_a.y * w4_b.y * w3_c.y * u0_kp1_jp2_ip2.y;
					}


					{
						real2 u0_kp2_jm1_im2;
						u0_kp2_jm1_im2 = _A(u0, k+2, j-1, (i-2) * !!i);
						real2 u0_kp2_jm1_i = _A(u0, k+2, j-1, i);
						real2 u0_kp2_jm1_im1; u0_kp2_jm1_im1.x = u0_kp2_jm1_im2.y; u0_kp2_jm1_im1.y = u0_kp2_jm1_i.x;
						real2 u0_kp2_jm1_ip2 = _A(u0, k+2, j-1, i+2);
						real2 u0_kp2_jm1_ip1; u0_kp2_jm1_ip1.x = u0_kp2_jm1_i.y; u0_kp2_jm1_ip1.y = u0_kp2_jm1_ip2.x;

						u1_x += w1_a.x * w1_b.x * w4_c.x * u0_kp2_jm1_im1.x +
							w2_a.x * w1_b.x * w4_c.x * u0_kp2_jm1_i.x +
							w3_a.x * w1_b.x * w4_c.x * u0_kp2_jm1_ip1.x +
							w4_a.x * w1_b.x * w4_c.x * u0_kp2_jm1_ip2.x;

						u1_y += w1_a.y * w1_b.y * w4_c.y * u0_kp2_jm1_im1.y +
							w2_a.y * w1_b.y * w4_c.y * u0_kp2_jm1_i.y +
							w3_a.y * w1_b.y * w4_c.y * u0_kp2_jm1_ip1.y +
							w4_a.y * w1_b.y * w4_c.y * u0_kp2_jm1_ip2.y;
					}

					{
						real2 u0_kp2_j_im2;
						u0_kp2_j_im2 = _A(u0, k+2, j, (i-2) * !!i);
						real2 u0_kp2_j_i = _A(u0, k+2, j, i);
						real2 u0_kp2_j_im1; u0_kp2_j_im1.x = u0_kp2_j_im2.y; u0_kp2_j_im1.y = u0_kp2_j_i.x;
						real2 u0_kp2_j_ip2 = _A(u0, k+2, j, i+2);
						real2 u0_kp2_j_ip1; u0_kp2_j_ip1.x = u0_kp2_j_i.y; u0_kp2_j_ip1.y = u0_kp2_j_ip2.x;

						u1_x += w1_a.x * w2_b.x * w4_c.x * u0_kp2_j_im1.x +
							w2_a.x * w2_b.x * w4_c.x * u0_kp2_j_i.x +
							w3_a.x * w2_b.x * w4_c.x * u0_kp2_j_ip1.x +
							w4_a.x * w2_b.x * w4_c.x * u0_kp2_j_ip2.x;

						u1_y += w1_a.y * w2_b.y * w4_c.y * u0_kp2_j_im1.y +
							w2_a.y * w2_b.y * w4_c.y * u0_kp2_j_i.y +
							w3_a.y * w2_b.y * w4_c.y * u0_kp2_j_ip1.y +
							w4_a.y * w2_b.y * w4_c.y * u0_kp2_j_ip2.y;
					}

					{
						real2 u0_kp2_jp1_im2;
						u0_kp2_jp1_im2 = _A(u0, k+2, j+1, (i-2) * !!i);
						real2 u0_kp2_jp1_i = _A(u0, k+2, j+1, i);
						real2 u0_kp2_jp1_im1; u0_kp2_jp1_im1.x = u0_kp2_jp1_im2.y; u0_kp2_jp1_im1.y = u0_kp2_jp1_i.x;
						real2 u0_kp2_jp1_ip2 = _A(u0, k+2, j+1, i+2);
						real2 u0_kp2_jp1_ip1; u0_kp2_jp1_ip1.x = u0_kp2_jp1_i.y; u0_kp2_jp1_ip1.y = u0_kp2_jp1_ip2.x;

						u1_x += w1_a.x * w3_b.x * w4_c.x * u0_kp2_jp1_im1.x +
							w2_a.x * w3_b.x * w4_c.x * u0_kp2_jp1_i.x +
							w3_a.x * w3_b.x * w4_c.x * u0_kp2_jp1_ip1.x +
							w4_a.x * w3_b.x * w4_c.x * u0_kp2_jp1_ip2.x;

						u1_y += w1_a.y * w3_b.y * w4_c.y * u0_kp2_jp1_im1.y +
							w2_a.y * w3_b.y * w4_c.y * u0_kp2_jp1_i.y +
							w3_a.y * w3_b.y * w4_c.y * u0_kp2_jp1_ip1.y +
							w4_a.y * w3_b.y * w4_c.y * u0_kp2_jp1_ip2.y;
					}

					{
						real2 u0_kp2_jp2_im2;
						u0_kp2_jp2_im2 = _A(u0, k+2, j+2, (i-2) * !!i);
						real2 u0_kp2_jp2_i = _A(u0, k+2, j+2, i);
						real2 u0_kp2_jp2_im1; u0_kp2_jp2_im1.x = u0_kp2_jp2_im2.y; u0_kp2_jp2_im1.y = u0_kp2_jp2_i.x;
						real2 u0_kp2_jp2_ip2 = _A(u0, k+2, j+2, i+2);
						real2 u0_kp2_jp2_ip1; u0_kp2_jp2_ip1.x = u0_kp2_jp2_i.y; u0_kp2_jp2_ip1.y = u0_kp2_jp2_ip2.x;		

						u1_x += w1_a.x * w4_b.x * w4_c.x * u0_kp2_jp2_im1.x +
							w2_a.x * w4_b.x * w4_c.x * u0_kp2_jp2_i.x +
							w3_a.x * w4_b.x * w4_c.x * u0_kp2_jp2_ip1.x +
							w4_a.x * w4_b.x * w4_c.x * u0_kp2_jp2_ip2.x;

						u1_y += w1_a.y * w4_b.y * w4_c.y * u0_kp2_jp2_im1.y +
							w2_a.y * w4_b.y * w4_c.y * u0_kp2_jp2_i.y +
							w3_a.y * w4_b.y * w4_c.y * u0_kp2_jp2_ip1.y +
							w4_a.y * w4_b.y * w4_c.y * u0_kp2_jp2_ip2.y;
					}

					real2* u1_k_j_i = &_A(u1, k, j, i);

					(*u1_k_j_i).y = u1_y;				

					// For the first vector we store only the second element,
					// since "i" starts from 1.
					if (i == 0) continue;
					
					(*u1_k_j_i).x = u1_x;
#endif

#elif defined(__CUDA_SHUFFLE__)

#if 0 // BEGIN TODO
					real val = _A(w1, k, j, i+2);
					real result = c2 * val;
					val = __shfl_up(val, 1);
					if (laneid == 0)
						val = _A(w1, k, j, i+1);
					result += c1 * val;
					val = __shfl_up(val, 1);
					if ((laneid == 0) || (laneid == 1))
						val = _A(w1, k, j, i);
					result += c0 * val;
					val = __shfl_up(val, 1);
					if ((laneid == 0) || (laneid == 1) || (laneid == 2))
						val = _A(w1, k, j, i-1);
					result += c1 * val;
					val = __shfl_up(val, 1);
					if ((laneid == 0) || (laneid == 1) || (laneid == 2) || (laneid == 3))
						val = _A(w1, k, j, i-2);
					result += c2 * val;

					if ((i < 2) || (i >= nx - 2)) continue;

					_A(w2, k, j, i) = result - _A(w0, k, j, i) +

						c1 * (
							_A(w1, k, j+1, i) + _A(w1, k, j-1, i)  +
							_A(w1, k+1, j, i) + _A(w1, k-1, j, i)) +
						c2 * (
							_A(w1, k, j+2, i) + _A(w1, k, j-2, i)  +
							_A(w1, k+2, j, i) + _A(w1, k-2, j, i));
#endif // END TODO

#else

					real w1_a;
					real w2_a;
					real w3_a;
					real w4_a;
					real w1_b;
					real w2_b;
					real w3_b;
					real w4_b;
					real w1_c;
					real w2_c;
					real w3_c;
					real w4_c;

					// This branch should be eliminated in compile-time.
					if (sizeof(real) == sizeof(float))
					{
						w1_a =  1.0f/6.0f                      * _A(a,k,j,i) * (_A(a,k,j,i)+1.0f) * (_A(a,k,j,i)+2.0f);
						w2_a = -0.5f      * (_A(a,k,j,i)-1.0f)               * (_A(a,k,j,i)+1.0f) * (_A(a,k,j,i)+2.0f);
						w3_a =  0.5f      * (_A(a,k,j,i)-1.0f) * _A(a,k,j,i)                      * (_A(a,k,j,i)+2.0f);
						w4_a = -1.0f/6.0f * (_A(a,k,j,i)-1.0f) * _A(a,k,j,i) * (_A(a,k,j,i)+1.0f);

						w1_b =  1.0f/6.0f                      * _A(b,k,j,i) * (_A(b,k,j,i)+1.0f) * (_A(b,k,j,i)+2.0f);
						w2_b = -0.5f      * (_A(b,k,j,i)-1.0f)               * (_A(b,k,j,i)+1.0f) * (_A(b,k,j,i)+2.0f);
						w3_b =  0.5f      * (_A(b,k,j,i)-1.0f) * _A(b,k,j,i)                      * (_A(b,k,j,i)+2.0f);
						w4_b = -1.0f/6.0f * (_A(b,k,j,i)-1.0f) * _A(b,k,j,i) * (_A(b,k,j,i)+1.0f);

						w1_c =  1.0f/6.0f                      * _A(c,k,j,i) * (_A(c,k,j,i)+1.0f) * (_A(c,k,j,i)+2.0f);
						w2_c = -0.5f      * (_A(c,k,j,i)-1.0f)               * (_A(c,k,j,i)+1.0f) * (_A(c,k,j,i)+2.0f);
						w3_c =  0.5f      * (_A(c,k,j,i)-1.0f) * _A(c,k,j,i)                      * (_A(c,k,j,i)+2.0f);
						w4_c = -1.0f/6.0f * (_A(c,k,j,i)-1.0f) * _A(c,k,j,i) * (_A(c,k,j,i)+1.0f);
					}
					else
					{
						w1_a =  1.0 /6.0                       * _A(a,k,j,i) * (_A(a,k,j,i)+1.0 ) * (_A(a,k,j,i)+2.0 );
						w2_a = -0.5       * (_A(a,k,j,i)-1.0 )               * (_A(a,k,j,i)+1.0 ) * (_A(a,k,j,i)+2.0 );
						w3_a =  0.5       * (_A(a,k,j,i)-1.0 ) * _A(a,k,j,i)                      * (_A(a,k,j,i)+2.0 );
						w4_a = -1.0 /6.0  * (_A(a,k,j,i)-1.0 ) * _A(a,k,j,i) * (_A(a,k,j,i)+1.0 );

						w1_b =  1.0 /6.0                       * _A(b,k,j,i) * (_A(b,k,j,i)+1.0 ) * (_A(b,k,j,i)+2.0 );
						w2_b = -0.5       * (_A(b,k,j,i)-1.0 )               * (_A(b,k,j,i)+1.0 ) * (_A(b,k,j,i)+2.0 );
						w3_b =  0.5       * (_A(b,k,j,i)-1.0 ) * _A(b,k,j,i)                      * (_A(b,k,j,i)+2.0 );
						w4_b = -1.0 /6.0  * (_A(b,k,j,i)-1.0 ) * _A(b,k,j,i) * (_A(b,k,j,i)+1.0 );

						w1_c =  1.0 /6.0                       * _A(c,k,j,i) * (_A(c,k,j,i)+1.0 ) * (_A(c,k,j,i)+2.0 );
						w2_c = -0.5       * (_A(c,k,j,i)-1.0 )               * (_A(c,k,j,i)+1.0 ) * (_A(c,k,j,i)+2.0 );
						w3_c =  0.5       * (_A(c,k,j,i)-1.0 ) * _A(c,k,j,i)                      * (_A(c,k,j,i)+2.0 );
						w4_c = -1.0 /6.0  * (_A(c,k,j,i)-1.0 ) * _A(c,k,j,i) * (_A(c,k,j,i)+1.0 );
					}
					
					_A(u1, k,j,i) =
				
						w1_a * w1_b * w1_c * _A(u0, k-1, j-1, i-1) +
						w2_a * w1_b * w1_c * _A(u0, k-1, j-1, i  ) +
						w3_a * w1_b * w1_c * _A(u0, k-1, j-1, i+1) +
						w4_a * w1_b * w1_c * _A(u0, k-1, j-1, i+2) +
			
						w1_a * w2_b * w1_c * _A(u0, k-1, j  , i-1) +
						w2_a * w2_b * w1_c * _A(u0, k-1, j  , i  ) +
						w3_a * w2_b * w1_c * _A(u0, k-1, j  , i+1) +
						w4_a * w2_b * w1_c * _A(u0, k-1, j  , i+2) +

						w1_a * w3_b * w1_c * _A(u0, k-1, j+1, i-1) +
						w2_a * w3_b * w1_c * _A(u0, k-1, j+1, i  ) +
						w3_a * w3_b * w1_c * _A(u0, k-1, j+1, i+1) +
						w4_a * w3_b * w1_c * _A(u0, k-1, j+1, i+2) +

						w1_a * w4_b * w1_c * _A(u0, k-1, j+2, i-1) +
						w2_a * w4_b * w1_c * _A(u0, k-1, j+2, i  ) +
						w3_a * w4_b * w1_c * _A(u0, k-1, j+2, i+1) +
						w4_a * w4_b * w1_c * _A(u0, k-1, j+2, i+2) +


						w1_a * w1_b * w2_c * _A(u0, k  , j-1, i-1) +
						w2_a * w1_b * w2_c * _A(u0, k  , j-1, i  ) +
						w3_a * w1_b * w2_c * _A(u0, k  , j-1, i+1) +
						w4_a * w1_b * w2_c * _A(u0, k  , j-1, i+2) +
			
						w1_a * w2_b * w2_c * _A(u0, k  , j  , i-1) +
						w2_a * w2_b * w2_c * _A(u0, k  , j  , i  ) +
						w3_a * w2_b * w2_c * _A(u0, k  , j  , i+1) +
						w4_a * w2_b * w2_c * _A(u0, k  , j  , i+2) +

						w1_a * w3_b * w2_c * _A(u0, k  , j+1, i-1) +
						w2_a * w3_b * w2_c * _A(u0, k  , j+1, i  ) +
						w3_a * w3_b * w2_c * _A(u0, k  , j+1, i+1) +
						w4_a * w3_b * w2_c * _A(u0, k  , j+1, i+2) +

						w1_a * w4_b * w2_c * _A(u0, k  , j+2, i-1) +
						w2_a * w4_b * w2_c * _A(u0, k  , j+2, i  ) +
						w3_a * w4_b * w2_c * _A(u0, k  , j+2, i+1) +
						w4_a * w4_b * w2_c * _A(u0, k  , j+2, i+2) +


						w1_a * w1_b * w3_c * _A(u0, k+1, j-1, i-1) +
						w2_a * w1_b * w3_c * _A(u0, k+1, j-1, i  ) +
						w3_a * w1_b * w3_c * _A(u0, k+1, j-1, i+1) +
						w4_a * w1_b * w3_c * _A(u0, k+1, j-1, i+2) +
			
						w1_a * w2_b * w3_c * _A(u0, k+1, j  , i-1) +
						w2_a * w2_b * w3_c * _A(u0, k+1, j  , i  ) +
						w3_a * w2_b * w3_c * _A(u0, k+1, j  , i+1) +
						w4_a * w2_b * w3_c * _A(u0, k+1, j  , i+2) +

						w1_a * w3_b * w3_c * _A(u0, k+1, j+1, i-1) +
						w2_a * w3_b * w3_c * _A(u0, k+1, j+1, i  ) +
						w3_a * w3_b * w3_c * _A(u0, k+1, j+1, i+1) +
						w4_a * w3_b * w3_c * _A(u0, k+1, j+1, i+2) +

						w1_a * w4_b * w3_c * _A(u0, k+1, j+2, i-1) +
						w2_a * w4_b * w3_c * _A(u0, k+1, j+2, i  ) +
						w3_a * w4_b * w3_c * _A(u0, k+1, j+2, i+1) +
						w4_a * w4_b * w3_c * _A(u0, k+1, j+2, i+2) +


						w1_a * w1_b * w4_c * _A(u0, k+2, j-1, i-1) +
						w2_a * w1_b * w4_c * _A(u0, k+2, j-1, i  ) +
						w3_a * w1_b * w4_c * _A(u0, k+2, j-1, i+1) +
						w4_a * w1_b * w4_c * _A(u0, k+2, j-1, i+2) +
			
						w1_a * w2_b * w4_c * _A(u0, k+2, j  , i-1) +
						w2_a * w2_b * w4_c * _A(u0, k+2, j  , i  ) +
						w3_a * w2_b * w4_c * _A(u0, k+2, j  , i+1) +
						w4_a * w2_b * w4_c * _A(u0, k+2, j  , i+2) +

						w1_a * w3_b * w4_c * _A(u0, k+2, j+1, i-1) +
						w2_a * w3_b * w4_c * _A(u0, k+2, j+1, i  ) +
						w3_a * w3_b * w4_c * _A(u0, k+2, j+1, i+1) +
						w4_a * w3_b * w4_c * _A(u0, k+2, j+1, i+2) +

						w1_a * w4_b * w4_c * _A(u0, k+2, j+2, i-1) +
						w2_a * w4_b * w4_c * _A(u0, k+2, j+2, i  ) +
						w3_a * w4_b * w4_c * _A(u0, k+2, j+2, i+1) +
						w4_a * w4_b * w4_c * _A(u0, k+2, j+2, i+2);

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
#if defined(_MIC) || defined(_OPENACC) || (defined(__CUDACC__) && !defined(_PPCG))
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
#if defined(__CUDACC__) && !defined(_PPCG)
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
#if defined(__CUDACC__) && !defined(_PPCG)
	get_time(&alloc_s);
	real *u0_dev = NULL, *u1_dev = NULL, *a_dev = NULL, *b_dev = NULL, *c_dev = NULL;
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
#if defined(_MIC) || defined(_OPENACC) || (defined(__CUDACC__) && !defined(_PPCG))
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
#if defined(__CUDACC__) && !defined(_PPCG)
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
#if defined(__CUDACC__) && !defined(_PPCG)
	dim3 gridDim, blockDim, strideDim;
	kernelgen_cuda_config_t config;
	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
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
		(config.blockDim.x + STENCIL_BOUNDARY_LEFT + STENCIL_BOUNDARY_RIGHT + SHMEM_BANK_SHIFT) *
		(config.blockDim.y + STENCIL_BOUNDARY_TOP + STENCIL_BOUNDARY_BOTTOM), STENCIL_BOUNDARY_TOP *
		(config.blockDim.x + STENCIL_BOUNDARY_LEFT + STENCIL_BOUNDARY_RIGHT + SHMEM_BANK_SHIFT) +
		STENCIL_BOUNDARY_LEFT);
#else
	kernelgen_cuda_configure_gird(1, nx, ny, ns, &config);
#endif
#endif
	get_time(&compute_s);
#if defined(_MIC)
	#pragma offload target(mic) \
		nocopy(u0:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(u1:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(a:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(b:length(szarray) alloc_if(0) free_if(0)), \
		nocopy(c:length(szarray) alloc_if(0) free_if(0))
#endif
	{
#if !defined(__CUDACC__) || defined(_PPCG)
		real *u0p = u0, *u1p = u1;
#else
		real *u0p = u0_dev, *u1p = u1_dev;
#endif
		for (int it = 0; it < nt; it++)
		{
#if !defined(__CUDACC__) || defined(_PPCG)
			tricubic(nx, ny, ns, u0p, u1p, a, b, c);
#else
			tricubic<<<config.gridDim, config.blockDim, config.szshmem>>>(nx, ny, ns,
				config,
				u0p, u1p, a_dev, b_dev, c_dev);
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

#if !defined(__CUDACC__) || defined(_PPCG)
	real* u[] = { u0, u1 }; 
	u0 = u[idxs[0]]; u1 = u[idxs[1]];
#else
	real* u[] = { u0_dev, u1_dev }; 
	u0_dev = u[idxs[0]]; u1_dev = u[idxs[1]];
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
		out(u1:length(szarray) alloc_if(0) free_if(0))
	{ }
	get_time(&save_f);
#endif
#if defined(_OPENACC)
	get_time(&save_s);
	#pragma acc update host (u1[0:szarray])
	get_time(&save_f);
#endif
#if defined(__CUDACC__) && !defined(_PPCG)
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
#if defined(_MIC) || (defined(__CUDACC__) && !defined(_PPCG))
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
#if defined(__CUDACC__) && !defined(_PPCG)
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

