!!===----------------------------------------------------------------------===//
!!
!!     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
!!        compiler for NVIDIA GPUs, targeting numerical modeling code.
!!
!! This file is distributed under the University of Illinois Open Source
!! License. See LICENSE.TXT for details.
!!
!!===----------------------------------------------------------------------===//

!dir$attributes offload:<MIC> :: jacobi_
#if defined(__CUDAFOR__)
attributes(global) &
#endif
subroutine jacobi(nx, ny, &
#if defined(__CUDAFOR__)
  i_stride, j_stride, &
#endif
  c0, c1, c2, w0, w1)

  implicit none

#if defined(__CUDAFOR__)
  integer, intent(in), value :: nx, ny
  integer, intent(in), value :: i_stride, j_stride
  real, intent(in), value :: c0, c1, c2
  real, intent(in), dimension(nx, ny), device :: w0
  real, intent(out), dimension(nx, ny), device :: w1
#else
  integer, intent(in) :: nx, ny
  real, intent(in) :: c0, c1, c2
  real, intent(in), dimension(nx, ny) :: w0
  real, intent(out), dimension(nx, ny) :: w1
#endif

  integer :: i, j

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
  !$acc kernels loop independent gang(65535), present(w0(1:nx,1:ny), w1(1:nx,1:ny))
#endif
#if defined(_OPENMP) || defined(_MIC)
  !$omp parallel do
#endif
  do j = 2 + j_offset, ny - 1, j_increment
#if defined(_OPENACC)
    !$acc loop independent vector(512)
#endif
    do i = 2 + i_offset, nx - 1, i_increment
      w1(i, j) = c0 *  w0(i,   j  ) + &
                 c1 * (w0(i-1, j  ) + w0(i  , j-1) + w0(i+1, j  ) + w0(i  , j+1)) + &
                 c2 * (w0(i-1, j-1) + w0(i-1, j+1) + w0(i+1, j-1) + w0(i+1, j+1))
    enddo
  enddo
  
end subroutine jacobi

