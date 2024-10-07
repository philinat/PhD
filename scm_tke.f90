MODULE scm_tke
  !!============================================================================<br />
  !!                       ***  MODULE  scm_tke  ***                            <br />
  !! Eddy-diffusion closure: TKE scheme with diagnostic mixing lengths
  !!                         adapted from NEMO TKE turbulent closure model      <br />
  !!============================================================================<br />
  !!----------------------------------------------------------------------------<br />
  !!   compute_tke_bdy   : top and bottom boundary conditions for TKE           <br />
  !!   compute_shear     : compute shear production term                        <br />
  !!   advance_tke       : tke time stepping: advance tke at next time step     <br />
  !!   compute_mxl       : compute mixing length scale                          <br />
  !!   compute_ED        : compute avm and avt                                  <br />
  !!   tridiag_solve_tke : tridiagonal solver for TKE equation                  <br />
  !!----------------------------------------------------------------------------<br />
    IMPLICIT NONE

CONTAINS

  SUBROUTINE compute_tke_bdy(taux, tauy, tke_const, bc_ap, wp0, tkemin, tke_sfc, tke_bot, flux_sfc)
    !!==========================================================================<br />
    !!                  ***  ROUTINE compute_tke_bdy  ***                       <br />
    !! ** Purposes : compute top and bottom boundary conditions for TKE equation<br />
    !!==========================================================================<br />
    USE scm_par, ONLY: cm_nemo,ceps_nemo,cm_mnh,ceps_mnh,cm_r81,ceps_r81
    IMPLICIT NONE
    REAL(8), INTENT(IN   )         :: taux         !! zonal surface stress      [m2/s2]
    REAL(8), INTENT(IN   )         :: tauy         !! meridional surface stress [m2/s2]
!    REAL(8), INTENT(IN   )         :: Bsfc         !! surface buoyancy flux     [m2/s3]
    INTEGER, INTENT(IN   )         :: tke_const    !! choice of TKE constants
    REAL(8), INTENT(IN   )         :: bc_ap        !! choice of TKE constants
    REAL(8), INTENT(IN   )         :: wp0          !! surface value for plume vertical velocity [m/s]
    REAL(8), INTENT(IN   )         :: tkemin
    REAL(8), INTENT(  OUT)         :: tke_sfc      !! surface value for Dirichlet condition [m2/s2]
    REAL(8), INTENT(  OUT)         :: tke_bot      !! bottom value for Dirichlet condition [m2/s2]
    REAL(8), INTENT(  OUT)         :: flux_sfc     !! surface TKE ED flux for Neumann condition [m3/s3]
    !local variables
    REAL(8)                        :: wstar2, ustar2, cff
    ! velocity scales
    !wstar2   =  ( -1. * MIN(Bsfc,0.) )**(2./3.)
    ustar2   =  SQRT( taux**2+tauy**2 )
    !
    IF( ustar2 == 0. ) THEN !! free convection case ( \( {\rm tke\_sfc\_dirichlet = True}  \) ) :
      tke_sfc  = 0.0001     !! \( k_{\rm sfc} = 1.5 \times 10^{-3}\;{\rm m}^2\;{\rm s}^{-2} \)<br />
      flux_sfc = 0.5*bc_ap*(wp0)**3 !! energetically consistent boundary condition \( F_{\rm sfc}^k = \left.  K_e \partial_z k \right|_{\rm sfc} \)
      !flux_sfc =0.
    ELSE
      IF(tke_const==0) THEN
        cff = 1./SQRT(cm_nemo*ceps_nemo)
      ELSE IF(tke_const==1) THEN
        cff = 1./SQRT( cm_mnh*ceps_mnh )
      ELSE
        cff = 1./SQRT( cm_r81*ceps_r81 )
      ENDIF
      tke_sfc  = cff*ustar2
      !tke_sfc  = 67.83*ustar2
      flux_sfc = 0.5*bc_ap*(wp0)**3 !! energetically consistent boundary condition \( F_{\rm sfc}^k = \left.  K_e \partial_z k \right|_{\rm sfc} \)
      !2.*67.83*ustar2*SQRT(ustar2)
    ENDIF
    ! bottom boundary condition
    tke_bot  = tkemin  !! bottom boundary condition : \( k_{\rm bot} = k_{\rm min}  \)
  !-----------------------------------------------------------------------------
  END SUBROUTINE compute_tke_bdy
  !=============================================================================

  SUBROUTINE compute_shear(u_n, v_n, u_np1, v_np1, Akv, zr, N, shear2)
    !!==========================================================================<br />
    !!                  ***  ROUTINE compute_shear  ***                         <br />
    !! ** Purposes : compute shear production term for TKE equation             <br />
    !!==========================================================================<br />
    IMPLICIT NONE
    INTEGER, INTENT(IN   )              :: N                      !! number of vertical levels
    REAL(8), INTENT(IN   )              :: u_n  (1:N),v_n  (1:N)  !! velocity components at time n    [m/s]
    REAL(8), INTENT(IN   )              :: u_np1(1:N),v_np1(1:N)  !! velocity components at time n+1  [m/s]
    REAL(8), INTENT(IN   )              :: zr(1:N)                !! depth at cell centers [m]
    REAL(8), INTENT(IN   )              :: Akv(0:N)               !! eddy-viscosity [m2/s]
    REAL(8), INTENT(  OUT)              :: shear2(0:N)            !! shear production term [m2/s3]
    ! local variables
    INTEGER                             :: k
    REAL(8)                             :: du,dv,cff
    shear2(0:N) = 0.
    DO k=1,N-1
      cff       = Akv(k) / ( zr(k+1)-zr(k) )**2
      du        = cff*( u_np1(k+1)-u_np1(k) )*0.5*( u_n(k+1)+u_np1(k+1)-u_n(k)-u_np1(k) ) !! Shear production term using discretization from Burchard (2002) <br />
      !! \( {\rm Sh}_{k+1/2} = \frac{ (K_m)_{k+1/2} }{ \Delta z_{k+1/2}^2 } ( u_{k+1}^n - u_{k}^n ) ( u_{k+1}^{n+1/2} - u_{k}^{n+1/2} )  \)
      dv        = cff*( v_np1(k+1)-v_np1(k) )*0.5*( v_n(k+1)+v_np1(k+1)-v_n(k)-v_np1(k) )
      shear2(k) = du + dv
    ENDDO
  !---------------------------------------------------------------------------------------------------
  END SUBROUTINE compute_shear
  !===================================================================================================
  !
  SUBROUTINE advance_tke( tke_n, lup, ldwn, Akv, Akt, Hz, zr, bvf, buoyMF, shear2,       &
                          shear2MF, trpl_corrMF, wtke, dt, tke_sfc, tke_bot, flux_sfc,   &
                          dirichlet_bdy_sfc, tke_const, tkemin, N, tke_np1, pdlr, eps, residual )
    !!==========================================================================<br />
    !!                  ***  ROUTINE advance_tke  ***                           <br />
    !! ** Purposes : tke time stepping, advance tke from time step n to n+1     <br />
    !!==========================================================================<br />
    USE scm_par, ONLY:ceps_nemo,Ric_nemo,ce_nemo,cm_nemo, &
                      ceps_mnh,Ric_mnh,ce_mnh,cm_mnh,ct_mnh, &
                      ceps_r81,Ric_r81,ce_r81,cm_r81,ct_r81,bshear,pdlrmin
    IMPLICIT NONE
    INTEGER, INTENT(IN   )   :: N                            !! number of vertical levels
    REAL(8), INTENT(IN   )   :: dt                           !! time-step [s]
    REAL(8), INTENT(IN   )   :: tke_sfc                      !! surface boundary condition for TKE [m2/s2]
    REAL(8), INTENT(IN   )   :: tke_bot                      !! bottom boundary condition for TKE [m2/s2]
    REAL(8), INTENT(IN   )   :: flux_sfc                     !! surface TKE ED flux for Neumann condition [m3/s3]
    REAL(8), INTENT(IN   )   :: tke_n  (0:N)                 !! TKE at time n    [m2/s2]
    REAL(8), INTENT(IN   )   :: lup    (0:N)                 !! upward mixing length [m]
    REAL(8), INTENT(IN   )   :: ldwn   (0:N)                 !! downward mixing length [m]
    REAL(8), INTENT(IN   )   :: Akt    (0:N)                 !! eddy-diffusion [m2/s]
    REAL(8), INTENT(IN   )   :: bvf    (0:N)                 !! Brunt Vaisala frequency [s-2]
    REAL(8), INTENT(IN   )   :: shear2 (0:N)                 !! shear tke production term [m2/s3]
    REAL(8), INTENT(IN   )   :: Akv    (0:N)                 !! eddy-viscosity [m2/s]
    REAL(8), INTENT(IN   )   :: Hz     (1:N)                 !! layer thickness [m]
    REAL(8), INTENT(IN   )   :: zr     (1:N)                 !! depth at cell centers [m]
    REAL(8), INTENT(IN   )   :: buoyMF (0:N)                 !! TKE buoyancy forcing term associated with mass flux [m2/s3]
    REAL(8), INTENT(IN   )   :: shear2MF   (0:N)             !! TKE shear forcing term associated with mass flux [m2/s3]
    REAL(8), INTENT(IN   )   :: trpl_corrMF(0:N)             !! Contribution of mass flux to d(w'e)/dz term [m2/s3]
    INTEGER, INTENT(IN   )   :: tke_const                    !! choice of TKE constants
    LOGICAL, INTENT(IN   )   :: dirichlet_bdy_sfc            !! Nature of the TKE surface boundary condition (T:dirichlet,F:Neumann)
    REAL(8), INTENT(IN   )   :: tkemin
    REAL(8), INTENT(INOUT)   :: wtke(1:N)                    !! Diagnostics : w'e term  [m3/s3]
    REAL(8), INTENT(  OUT)   :: tke_np1(0:N)                 !! TKE at time n+1    [m2/s2]
    REAL(8), INTENT(  OUT)   :: pdlr(0:N)                    !! inverse of turbulent Prandtl number
    REAL(8), INTENT(  OUT)   :: eps(0:N)                     !! TKE dissipation term [m2/s3]
    REAL(8), INTENT(  OUT)   :: residual                     !! Diagnostics : TKE spuriously added to guarantee that tke >= tke_min [m3/s3]
    ! local variables
    INTEGER                  :: k
    REAL(8)                  :: mxld(0:N),mxlm(0:N),ff(0:N), cff, Ric, isch
    REAL(8)                  :: sh2,buoy,Ri,cff1,cff2,cff3,rhs,rhsmin,ceps,ct,cm
    ! Initialization
    tke_np1(  N  ) = tkemin
    IF(dirichlet_bdy_sfc) tke_np1(  N  ) = tke_sfc
    !
    tke_np1(  0  ) = tke_bot
    tke_np1(1:N-1) = tkemin
    ! Initialize constants
    IF(tke_const==0) THEN
      ceps = ceps_nemo; Ric  = Ric_nemo; isch = ce_nemo / cm_nemo
    ELSE IF(tke_const==1) THEN
      cm   = cm_mnh; ct   = ct_mnh; ceps = ceps_mnh
      Ric  = Ric_mnh; isch = ce_mnh / cm_mnh
    ELSE
      cm   = cm_r81; ct   = ct_r81; ceps = ceps_r81
      Ric  = Ric_r81; isch = ce_r81 / cm_r81
    ENDIF
    !
    DO k=1,N-1
      mxld(k) = SQRT( lup(k) * ldwn(k) ) !! Dissipative mixing length : \( (l_\epsilon)_{k+1/2} = \sqrt{ l_{\rm up} l_{\rm dwn} }   \)  <br />
      mxlm(k) = MIN ( lup(k),  ldwn(k) )
    ENDDO
    !====================================================================
    pdlr(:) = 0. !! Inverse Prandtl number function of Richardson number <br />
    !IF(tke_const==0) THEN
      DO k = 1,N-1
        sh2     = shear2(k) ! shear2 is already multiplied by Akv
        buoy    = bvf(k)
        Ri      = MAX( buoy, 0. ) * Akv(k) / ( sh2 + bshear )  !! \( {\rm Ri}_{k+1/2} = (K_m)_{k+1/2} (N^2)_{k+1/2} / {\rm Sh}_{k+1/2} \) <br />
        pdlr(k) = MAX(  pdlrmin,  Ric / MAX( Ric , Ri ) )      !! \( ({\rm Pr}_t)^{-1}_{k+1/2} = \max\left( {\rm Pr}_{\min}^{-1} , \frac{{\rm Ri}_c}{ \max( {\rm Ri}_c, {\rm Ri}_{k+1/2}  ) } \right)     \) <br />
      END DO
    !ELSE
    !  DO k = 1,N-1
    !    buoy    = bvf(k)
    !    Ri      = buoy*mxld(k)*mxlm(k)/tke_n(k)  ! Redelsperger number
    !    pdlr(k) = MAX( pdlrmin, ct/(cm+cm*MAX(-0.5455, Ric*Ri ) ) )
    !  END DO
    !ENDIF
    !====================================================================
    ! constants for TKE dissipation term
    cff1   =  0.5; cff2   =  1.5; cff3   =  cff1/cff2
    !
    eps(0:N) = 0.
    ff (0:N) = 0.
    residual = 0.
    !
    DO k = 1,N-1
      ! construct the right hand side
      rhs       = shear2(k) - Akt(k) * bvf(k)   &
                           + shear2MF(k) +  buoyMF(k) + trpl_corrMF(k) !! \(  {\rm rhs}_{k+1/2} = {\rm Sh}_{k+1/2} - (K_s N^2)_{k+1/2} + {\rm Sh}_{k+1/2}^{\rm p} + (-a^{\rm p} w^{\rm p} B^{\rm p})_{k+1/2} + {\rm TOM}_{k+1/2}   \) <br />
      ! dissipation divided by tke
      eps(k)    = cff2*ceps*SQRT(tke_n(k))/mxld(k)
      ! increase rhs if too small to guarantee that tke > tke_min
      rhsmin    = (tkemin-tke_n(k))/dt   &
                           + eps(k)*tkemin - cff3*eps(k)*tke_n(k)
      ! right hand side for tridiagonal problem
      ff(k)     = tke_n(k) + dt*MAX(rhs,rhsmin) + dt*cff3*eps(k)*tke_n(k)  !! Right-hand-side for tridiagonal problem \( f_{k+1/2} = k^n_{k+1/2} + \Delta t {\rm rhs}_{k+1/2} + \frac{1}{2} \Delta t c_\epsilon \frac{ k^n_{k+1/2} \sqrt{k^n_{k+1/2}} }{(l_\epsilon)_{k+1/2}}   \)<br />
      ! keep track of the energy spuriously added to get tke > tke_min
      IF(rhs<rhsmin) residual = residual + (zr(k+1)-zr(k))*(rhsmin-rhs)
    ENDDO
    !! Boundary conditions : <br />
    ff(0) = tke_bot
    IF(dirichlet_bdy_sfc) THEN    !! \( {\rm dirichlet\_bdy\_sfc = True}\qquad  \rightarrow \qquad k_{N+1/2}^{n+1} = k_{\rm sfc}  \)  <br />
      ff(N) = tke_sfc
    ELSE
      ff(N) = 2.*Hz(N)*flux_sfc / ( isch*(Akv(N)+Akv(N-1)) ) !! \( {\rm dirichlet\_bdy\_sfc = False}\qquad  \rightarrow \qquad k_{N+1/2}^{n+1} - k_{N+1/2}^{n} = 2 \frac{\Delta z_{N} F_{\rm sfc}^k}{(K_e)_{N+1/2}+ (K_e)_{N-1/2}}  \)  <br />
    ENDIF
    !
    CALL tridiag_solve_tke(N,Hz,isch*Akv,zr,eps,ff,dt,dirichlet_bdy_sfc)  !! Solve the tridiagonal problem
    !
    DO k = 0,N
      tke_np1(k)=MAX(ff(k),tkemin)
    ENDDO
    ! ** Diagnostics **
    !         Store the TKE dissipation term for diagnostics
    eps    (1:N-1)= ceps*(cff2*tke_np1(1:N-1)-cff1*tke_n(1:N-1))*(SQRT( tke_n(1:N-1) )/mxld(1:N-1))
    eps(0)=0.; eps(N)=0.
    !         Store the ED contribution to w'e turbulent flux
    DO k = 1,N
    wtke(k) = wtke(k) - 0.5*isch*(akv(k)+akv(k-1))*(tke_np1(k)-tke_np1(k-1))/Hz(k)
    ENDDO
  !---------------------------------------------------------------------------------------------------
  END SUBROUTINE advance_tke
  !===================================================================================================




  !===================================================================================================
  SUBROUTINE compute_mxl(tke, bvf, Hz, taux, tauy, mxlmin, N, lup, ldwn)
  !---------------------------------------------------------------------------------------------------
    !!============================================================================<br />
    !!                  ***  ROUTINE compute_mxl  ***                             <br />
    !! ** Purposes : compute mixing length scales                                 <br />
    !!============================================================================<br />
    USE scm_par, ONLY:grav,vkarmn,rsmall,mxl_min0
    IMPLICIT NONE
    INTEGER, INTENT(IN   )  :: N              !! number of vertical levels
    REAL(8), INTENT(IN   )  :: tke(0:N)       !! turbulent kinetic energy [m2/s2]
    REAL(8), INTENT(IN   )  :: bvf(0:N)       !! Brunt Vaisala frequency [s-2]
    REAL(8), INTENT(IN   )  :: Hz(1:N)        !! layer thickness [m]
    REAL(8), INTENT(IN   )  :: taux           !! surface stress [m2/s2]
    REAL(8), INTENT(IN   )  :: tauy           !! surface stress [m2/s2]
    REAL(8), INTENT(IN   )  :: mxlmin         !!
    REAL(8), INTENT(  OUT)  :: lup(0:N)       !! upward mixing length [m]
    REAL(8), INTENT(  OUT)  :: ldwn(0:N)      !! downward mixing length [m]
    ! local variables
    INTEGER                 :: k
    REAL(8)                 :: rn2,ld80(0:N),raug,ustar2
    !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    !  Mixing length
    !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ustar2     =  SQRT( taux**2+tauy**2 )  !! \( u_{\star}^2 = \sqrt{\tau_x^2 + \tau_y^2} \)<br />
    !
    lup (0:N)  = mxlmin
    ldwn(0:N)  = mxlmin
    !
    DO k = 0, N              ! interior value : l=sqrt(2*e/n^2)
      rn2     = MAX( bvf(k) , rsmall )
      ld80(k) = MAX( mxlmin,  SQRT(2.*tke(k) / rn2 ) ) !! Buoyancy length scale : \(  (l_{\rm up})_{k+1/2}=(l_{\rm dwn})_{k+1/2}=(l_{D80})_{k+1/2} = \sqrt{\frac{2 k_{k+1/2}^{n+1}}{ \max( (N^2)_{k+1/2}, (N^2)_{\min} ) }}     \)<br />
    END DO
    !! Physical limits for the mixing lengths <br />
    ldwn(0  ) = 0.
    DO k = 1, N
      ldwn(k) = MIN( ldwn(k-1) + Hz(k  ) , ld80(k) )   !! Limit \( (l_{\rm dwn})_{k+1/2} \) such that \( \partial_z (l_{\rm dwn})_{k} \le 1 \) the bottom boundary condition is \( (l_{\rm dwn})_{1/2} = l_{\min} \) <br />
    END DO
    ! surface mixing length = F(stress)=vkarmn*2.e5*taum/(rho0*g)
    raug      = vkarmn * 2.e5 / grav
    lup(N)   = MAX( mxl_min0, raug * ustar2 ) ! surface boundary condition
    !
    DO k = N-1,0,-1
      lup(k) = MIN( lup(k+1) + Hz(k+1) , ld80(k) )   !! Limit \( (l_{\rm up})_{k-1/2} \) such that \( - \partial_z (l_{\rm up})_{k} \le 1 \) the surface boundary condition is \( (l_{\rm up})_{N+1/2} = \frac{\kappa}{g} (2 \times 10^{-5}) u_{\star}^2 \) <br />
    END DO
    lup(N) = 0.   !<-- ensures that avm(jpk) = 0.
    !
  !---------------------------------------------------------------------------------------------------
  END SUBROUTINE compute_mxl
  !===================================================================================================

  !===================================================================================================
  SUBROUTINE compute_ED(tke,lup,ldwn,pdlr,extrap_sfc,tke_const,Akvmin,Aktmin,N,Akv,Akt)
  !---------------------------------------------------------------------------------------------------
    !!============================================================================<br />
    !!                  ***  ROUTINE compute_ED  ***                              <br />
    !! ** Purposes : compute the vertical eddy viscosity and diffusivity          <br />
    !!============================================================================<br />
    USE scm_par, ONLY: cm_nemo,cm_mnh,cm_r81
    IMPLICIT NONE
    INTEGER, INTENT(IN   )  :: N                   !! number of vertical levels
    REAL(8), INTENT(IN   )  :: tke(0:N)            !! turbulent kinetic energy [m2/s2]
    REAL(8), INTENT(IN   )  :: pdlr(0:N)           !! inverse turbulent Prandtl number
    REAL(8), INTENT(IN   )  :: lup(0:N)            !! upward mixing length [m]
    REAL(8), INTENT(IN   )  :: ldwn(0:N)           !! downward mixing length [m]
    REAL(8), INTENT(IN   )  :: Akvmin
    REAL(8), INTENT(IN   )  :: Aktmin
    LOGICAL, INTENT(IN   )  :: extrap_sfc          !! (T) extrapolate eddy coefficients to the surface
    INTEGER, INTENT(IN   )  :: tke_const           !! choice of TKE constants
    REAL(8), INTENT(  OUT)  :: Akv(0:N)            !! eddy-viscosity [m2/s]
    REAL(8), INTENT(  OUT)  :: Akt(0:N)            !! eddy-diffusivity [m2/s]
    ! local variables
    INTEGER                 :: k
    REAL(8)                 :: mxlm,av,cm,ct
    !* Initialize constants
    IF(tke_const==0) THEN
      cm = cm_nemo
    ELSE IF(tke_const==1) THEN
      cm = cm_mnh
    ELSE
      cm = cm_r81
    ENDIF
    !
    DO k = 0, N
      mxlm          = MIN ( lup(k),  ldwn(k) )      !! Compute "master" mixing length \( (l_m)_{k+1/2} = \min( (l_{\rm up})_{k+1/2}, (l_{\rm dwn})_{k+1/2} ) \)<br />
      av            = cm * mxlm * SQRT(tke(k))      !! Compute eddy-viscosity \( (K_m)_{k+1/2} = C_m l_m \sqrt{k_{k+1/2}^{n+1}}  \)<br />
      Akv  (k  )    = MAX(           av,Akvmin )
      Akt  (k  )    = MAX( pdlr(k) * av,Aktmin )   !! Compute eddy-diffusivity \( (K_s)_{k+1/2} = ({\rm Pr}_t)^{-1}_{k+1/2}   (K_m)_{k+1/2} \)<br />
    END DO
    Akv(N) = 0.; Akt(N) = 0.
    !Warning : extrapolation ignores the variations of Hz with depth
    IF(extrap_sfc) THEN
      Akv(N) = 1.5*Akv(N-1)-0.5*Akv(N-2) !! if \( {\rm extrap\_sfc = True} \qquad \rightarrow \qquad (K_m)_{N+1/2} = \frac{3}{2} (K_m)_{N-1/2} - \frac{1}{2} (K_m)_{N-3/2} \)
      Akt(N) = 1.5*Akt(N-1)-0.5*Akt(N-2)
    ENDIF
  !---------------------------------------------------------------------------------------------------
  END SUBROUTINE compute_ED
  !===================================================================================================
!
!#################################################################
! TRIDIAGONAL INVERSION
!#################################################################
  !===================================================================================================
  SUBROUTINE tridiag_solve_tke(N,Hz,Ak,zr,eps,f,dt,dirichlet_bdy_sfc)
  !---------------------------------------------------------------------------------------------------
    !!============================================================================<br />
    !!                  ***  ROUTINE tridiag_solve_tke  ***                       <br />
    !! ** Purposes : solve the tridiagonal problem associated with the implicit
    !!                                         in time treatment of TKE equation  <br />
    !!============================================================================<br />
    IMPLICIT NONE
    INTEGER, INTENT(IN   )      :: N                 !! number of vertical levels
    REAL(8), INTENT(IN   )      :: dt                !! time step [s]
    REAL(8), INTENT(IN   )      :: Hz(1:N)           !! layer thickness [m]
    REAL(8), INTENT(IN   )      :: Ak(0:N)           !! eddy-diffusivity for TKE [m2/s]
    REAL(8), INTENT(IN   )      :: zr(1:N)           !! depth at cell centers [m]
    REAL(8), INTENT(IN   )      :: eps(0:N)          !! TKE dissipation term divided by TKE [s-1]
    LOGICAL, INTENT(IN   )      :: dirichlet_bdy_sfc !! nature of the TKE boundary condition
    REAL(8), INTENT(INOUT)      :: f(0:N)            !! (in) rhs for tridiagonal problem (out) solution of the tridiagonal problem
    ! local variables
    INTEGER                     :: k
    REAL(8)                     :: a(0:N),b(0:N),c(0:N),q(0:N)
    REAL(8)                     :: difA,difC,cff
    !
    DO k=1,N-1
      difA  = -0.5*dt*(Ak(k-1)+Ak(k))/(Hz(k  )*(zr(k+1)-zr(k)))
      difC  = -0.5*dt*(Ak(k+1)+Ak(k))/(Hz(k+1)*(zr(k+1)-zr(k)))
      a (k) = difA
      c (k) = difC
      b (k) = 1. - difA - difC + dt*eps(k)
    ENDDO
    !++ Bottom BC
    a (0) = 0.; b(0) = 1; c(0) = 0.
    IF(dirichlet_bdy_sfc) THEN
      a (N) = 0.; b(N) = 1.; c(N) = 0.
    ELSE
      a (N) = -1.; b(N) = -a(N); c(N) = 0.
    ENDIF
    !
    cff   = 1./b(0)
    q (0) = - c(0)*cff
    f (0) =   f(0)*cff
    DO k=1,N
      cff=1./(b(k)+a(k)*q(k-1))
      q(k)= -cff*c(k)
      f(k)=cff*( f(k)-a(k)*f(k-1) )
    ENDDO
    DO k=N-1,0,-1
      f(k)=f(k)+q(k)*f(k+1)
    ENDDO
  !---------------------------------------------------------------------------------------------------
  END SUBROUTINE tridiag_solve_tke
  !===================================================================================================
!
END MODULE scm_tke
