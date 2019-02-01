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
subroutine sincos(nx, ny, ns, &
#if defined(__CUDAFOR__)
  i_stride, j_stride, k_stride, &
#endif
  x, y, xy)

  implicit none

#if defined(__CUDAFOR__)
  integer, intent(in), value :: nx, ny, ns
  integer, intent(in), value :: i_stride, j_stride, k_stride
  real, intent(in), dimension(nx, ny, ns), device :: x, y
  real, intent(out), dimension(nx, ny, ns), device :: xy
#else
  integer, intent(in) :: nx, ny, ns
  real, intent(in), dimension(nx, ny, ns) :: x, y
  real, intent(out), dimension(nx, ny, ns) :: xy
#endif
  integer :: i, j, k
  
#if defined(__CUDAFOR__)
  integer :: k_offset, j_offset, i_offset, k_increment, j_increment, i_increment
#else
  integer, parameter :: k_offset = 0
  integer, parameter :: j_offset = 0
  integer, parameter :: i_offset = 0
  integer, parameter :: k_increment = 1
  integer, parameter :: j_increment = 1
  integer, parameter :: i_increment = 1
#endif

#if defined(__CUDAFOR__)
  k_offset = (blockIdx%z - 1) * blockDim%z + threadIdx%z - 1
  j_offset = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
  i_offset = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
  k_increment = k_stride
  j_increment = j_stride
  i_increment = i_stride
#endif

#if defined(_OPENACC)
  !$acc kernels loop independent gang(65535), present(x(1:nx,1:ny,1:ns), y(1:nx,1:ny,1:ns), xy(1:nx,1:ny,1:ns))
#endif
#if defined(_OPENMP) || defined(_MIC)
  !$omp parallel for
#endif
  do k = 1 + k_offset, ns, k_increment
#if defined(_OPENACC)
    !$acc loop independent
#endif
    do j = 1 + j_offset, ny, j_increment
#if defined(_OPENACC)
      !$acc loop independent vector(512)
#endif
      do i = 1 + i_offset, nx, i_increment
        xy(i,j,k) = sin(x(i,j,k)) + cos(y(i,j,k))
      enddo
    enddo
  enddo

end subroutine sincos	

