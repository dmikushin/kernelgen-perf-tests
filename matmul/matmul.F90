!!===----------------------------------------------------------------------===//
!!
!!     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
!!        compiler for NVIDIA GPUs, targeting numerical modeling code.
!!
!! This file is distributed under the University of Illinois Open Source
!! License. See LICENSE.TXT for details.
!!
!!===----------------------------------------------------------------------===//

#if defined(__CUDAFOR__)
attributes(global) &
#endif
subroutine matmul(nx, ny, ns, &
#if defined(__CUDAFOR__)
  i_stride, j_stride, &
#endif
  A, B, C)

#if defined(__CUDAFOR__)
  integer, intent(in), value :: nx, ny, ns
  integer, intent(in), value :: i_stride, j_stride
  real, intent(in), dimension(nx, ny), device :: A
  real, intent(in), dimension(ny, ns), device :: B
  real, intent(inout), dimension(nx, ns), device :: C
#else
  integer, intent(in) :: nx, ny, ns
  real, intent(in), dimension(nx, ny) :: A
  real, intent(in), dimension(ny, ns) :: B
  real, intent(inout), dimension(nx, ns) :: C
#endif
  integer :: i, j, k

#if defined(__CUDAFOR__)
  integer :: j_offset, i_offset, j_increment, i_increment
#else
  integer, parameter :: j_offset = 0
  integer, parameter :: i_offset = 0
  integer, parameter :: j_increment = 1
  integer, parameter :: i_increment = 1
#endif

#if defined(__CUDAFOR__)
  j_offset = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
  i_offset = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
  j_increment = j_stride
  i_increment = i_stride
#endif

#if defined(_OPENACC)
  !$acc kernels loop gang(65535)
#endif
#if defined(_OPENMP) || defined(_MIC)
  !$omp parallel for
#endif
  do j = 1 + j_offset, ns, j_increment
#if defined(_OPENACC)
    !$acc loop vector(512)
#endif
    do i = 1 + i_offset, nx, i_increment
#if defined(_OPENACC)
      !$acc loop seq 
#endif
      do k = 1, ny
        C(i, j) = C(i, j) + A(i, k) * B(k, j)
      enddo
    enddo
  enddo

end subroutine matmul

