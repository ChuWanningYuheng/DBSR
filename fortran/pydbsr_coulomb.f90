!======================================================================
!     C-callable wrapper around A.R. Barnett's COULFG (ZCOM/FUN_coulfg.f)
!     used by the Python outer-region code (pydbsr.coulomb, via ctypes).
!
!     For i = 1..n returns the regular/irregular Coulomb functions
!     F_l(eta,rho), G_l(eta,rho) and their derivatives d/d(rho).
!     ifail(i) /= 0 signals failure of COULFG for that point.
!======================================================================
      Subroutine pydbsr_coulfg(n, rho, eta, l, f, g, fp, gp, ifail) &
                 bind(C, name='pydbsr_coulfg')
      Use, intrinsic :: iso_c_binding
      Implicit none
      Integer(c_int), value :: n
      Real(c_double), intent(in)  :: rho(n), eta(n)
      Integer(c_int), intent(in)  :: l(n)
      Real(c_double), intent(out) :: f(n), g(n), fp(n), gp(n)
      Integer(c_int), intent(out) :: ifail(n)
      Real(8), allocatable :: fc(:), gc(:), fcp(:), gcp(:)
      Real(8) :: xl, x0
      Integer :: i, m, ierr

      m = max(maxval(l), 0) + 2
      Allocate(fc(m), gc(m), fcp(m), gcp(m))
      Do i = 1, n
       fc = 0.d0; gc = 0.d0; fcp = 0.d0; gcp = 0.d0
       ! Recursion from lambda = 0.  COULFG evaluates its continued fraction
       ! CF1 at lambda = XLMAX; an accidental zero of CF1 gives wrong values
       ! (with the Wronskian still = 1), so compute with XLMAX = l and l+1
       ! and flag disagreement (the caller then uses another method).
       x0 = 0.d0
       xl = dble(l(i)+1)
       Call COULFG(rho(i), eta(i), x0, xl, fc, gc, fcp, gcp, 1, 0, ierr)
       f(i)  = fc(l(i)+1);  g(i)  = gc(l(i)+1)
       fp(i) = fcp(l(i)+1); gp(i) = gcp(l(i)+1)
       ifail(i) = ierr
       xl = dble(l(i))
       Call COULFG(rho(i), eta(i), x0, xl, fc, gc, fcp, gcp, 1, 0, ierr)
       if(ierr.ne.0) ifail(i) = ierr
       if(abs(fc(l(i)+1)-f(i)).gt.1.d-10*(abs(f(i))+abs(g(i))) .or. &
          abs(gc(l(i)+1)-g(i)).gt.1.d-10*(abs(f(i))+abs(g(i)))) ifail(i) = 99
      End do
      Deallocate(fc, gc, fcp, gcp)

      End Subroutine pydbsr_coulfg
