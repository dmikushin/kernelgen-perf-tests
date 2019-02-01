//===----------------------------------------------------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define CUDA_SAFE_CALL(x) \
	do { cudaError_t err = x; if (err != cudaSuccess) { \
		fprintf (stderr, "Error \"%s\" at %s:%d \n", cudaGetErrorString(err), \
		__FILE__, __LINE__); exit(-1); \
	}} while (0);

#ifdef __cplusplus
extern "C" {
#endif

int kernelgen_enable_regcount(char* funcname, long lineno);

int kernelgen_disable_regcount();

int kernelgen_cuda_configure_gird(int nx, int ny, int ns,
	dim3* gridDim, dim3* blockDim, dim3* strideDim);

#ifdef __cplusplus
}
#endif

