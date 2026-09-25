!======================================================================
      Module pydbsr_ilp64
!======================================================================
!     DSYEVD with 64-bit integers, loaded at run time (added by pydbsr).
!
!     LAPACK with 32-bit integers cannot run DSYEVD for n > ~32000: its
!     workspace (1 + 6n + 2n^2) overflows Integer(4).  dbsr_hd3 calls
!     pydbsr_dsyevd64() for such matrices; it dlopen()s the ILP64 LAPACK
!     given by $PYDBSR_ILP64_LAPACK (libscipy_openblas64_.so of the
!     scipy-openblas64 wheel, or conda's libopenblas64_.so) and calls its
!     DSYEVD.  Pure Fortran (no C compiler involved in linking).
!
!     Result: LAPACK info (0 = success); -9999 no ILP64 library,
!     -9998 no memory for the workspace.
!----------------------------------------------------------------------
      Use, intrinsic :: iso_c_binding
      Implicit none
      Private
      Public :: pydbsr_dsyevd64

      Interface
       Function c_dlopen(name, mode) bind(C, name='dlopen') result(h)
        Import :: c_char, c_int, c_ptr
        Character(kind=c_char), dimension(*) :: name
        Integer(c_int), value :: mode
        Type(c_ptr) :: h
       End Function c_dlopen
       Function c_dlsym(h, name) bind(C, name='dlsym') result(p)
        Import :: c_char, c_ptr, c_funptr
        Type(c_ptr), value :: h
        Character(kind=c_char), dimension(*) :: name
        Type(c_funptr) :: p
       End Function c_dlsym
      End Interface

      Abstract Interface
       Subroutine dsyevd64_t(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, &
                             ljobz, luplo) bind(C)
        Import :: c_char, c_int64_t, c_double, c_size_t
        Character(kind=c_char) :: jobz, uplo
        Integer(c_int64_t) :: n, lda, lwork, liwork, info
        Real(c_double) :: a(*), w(*), work(*)
        Integer(c_int64_t) :: iwork(*)
        Integer(c_size_t), value :: ljobz, luplo
       End Subroutine dsyevd64_t
      End Interface

      Contains

      Integer Function pydbsr_dsyevd64(jobz, uplo, n, a, w)
      Character(1), intent(in) :: jobz, uplo
      Integer, intent(in) :: n
      Real(8) :: a(*), w(*)
      Character(4096) :: path
      Character(len=20), dimension(2) :: names = [Character(len=20) :: 'scipy_dsyevd_64_', 'dsyevd_64_']
      Type(c_ptr) :: h
      Type(c_funptr) :: fp
      Procedure(dsyevd64_t), pointer :: f
      Integer(c_int64_t) :: n8, lwork, liwork, info, iwq(1)
      Real(c_double) :: wq(1)
      Real(c_double), allocatable :: work(:)
      Integer(c_int64_t), allocatable :: iwork(:)
      Character(kind=c_char) :: cj, cu
      Integer :: i, ierr

      pydbsr_dsyevd64 = -9999
      Call get_environment_variable('PYDBSR_ILP64_LAPACK', path, status=ierr)
      if(ierr.ne.0 .or. len_trim(path).eq.0) Return
      h = c_dlopen(trim(path)//c_null_char, 2_c_int)            ! RTLD_NOW
      if(.not.c_associated(h)) then
       write(*,*) 'pydbsr_dsyevd64: cannot load ', trim(path)
       Return
      end if
      fp = c_null_funptr
      Do i = 1, size(names)
       fp = c_dlsym(h, trim(names(i))//c_null_char)
       if(c_associated(fp)) Exit
      End do
      if(.not.c_associated(fp)) then
       write(*,*) 'pydbsr_dsyevd64: no ILP64 dsyevd in ', trim(path)
       Return
      end if
      Call c_f_procpointer(fp, f)

      cj = jobz; cu = uplo; n8 = n; info = 0
      lwork = -1; liwork = -1
      Call f(cj, cu, n8, a, n8, w, wq, lwork, iwq, liwork, info, 1_c_size_t, 1_c_size_t)
      if(info.ne.0) then
       pydbsr_dsyevd64 = int(info); Return
      end if
      lwork = int(wq(1), c_int64_t); liwork = iwq(1)
      Allocate(work(lwork), iwork(liwork), stat=ierr)
      if(ierr.ne.0) then
       pydbsr_dsyevd64 = -9998; Return
      end if
      Call f(cj, cu, n8, a, n8, w, work, lwork, iwork, liwork, info, 1_c_size_t, 1_c_size_t)
      Deallocate(work, iwork)
      pydbsr_dsyevd64 = int(info)
      End Function pydbsr_dsyevd64

      End Module pydbsr_ilp64
