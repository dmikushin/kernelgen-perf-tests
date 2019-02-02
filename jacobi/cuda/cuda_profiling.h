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

struct kernelgen_cuda_config_t
{
	dim3 gridDim, blockDim, strideDim;
	
	// The total size of shared memory requried
	// (to be used with the kernel launch call).
	size_t szshmem;
	
	// The array of offsets where the corresponding
	// shared memory scratch buffers of input data
	// arrays shall start.
	ptrdiff_t* shmem_arrays;
};

int kernelgen_cuda_configure_gird(int ndims, int nx, int ny, int ns,
	kernelgen_cuda_config_t* config);

int kernelgen_cuda_configure_shmem(kernelgen_cuda_config_t* config,
	int narrays, ...);

void kernelgen_cuda_config_dispose(kernelgen_cuda_config_t* config);

#ifdef __cplusplus
}
#endif

