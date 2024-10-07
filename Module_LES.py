#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:44:17 2024

@author: philippotn
"""
from numba import njit,prange
import numpy as np
floatype = np.float32
intype = np.int32

# Constants from mesoNH
AVO_BOLZ = 6.0221367E+23 * 1.380658e-23
Md    = 28.9644E-3
Mv    = 18.0153E-3
Rd = AVO_BOLZ / Md     # J/kg/K
Rv = AVO_BOLZ / Mv     # J/kg/K
Cp = 7/2 * Rd
Cpv = 4 * Rv
T0 = 273.16 # K
Lv = 2.5008E+6 # J/kg
g = 9.80665  # m/s^2 
p0 = 1e5 # Pa 
        
#%% Thermodynamics

# Version 1.0 released by David Romps on September 12, 2017.
# Version 1.1 vectorized lcl.R, released on May 24, 2021.
# 
# When using this code, please cite:
# 
# @article{16lcl,
#   Title   = {Exact expression for the lifting condensation level},
#   Author  = {David M. Romps},
#   Journal = {Journal of the Atmospheric Sciences},
#   Year    = {2017},
#   Month   = dec,
#   Number  = {12},
#   Pages   = {3891--3900},
#   Volume  = {74}
# }
#
# This lcl function returns the height of the lifting condensation level
# (LCL) in meters.  The inputs are:
# - p in Pascals
# - T in Kelvins
# - Exactly one of rh, rhl, and rhs (dimensionless, from 0 to 1):
#    * The value of rh is interpreted to be the relative humidity with
#      respect to liquid water if T >= 273.15 K and with respect to ice if
#      T < 273.15 K. 
#    * The value of rhl is interpreted to be the relative humidity with
#      respect to liquid water
#    * The value of rhs is interpreted to be the relative humidity with
#      respect to ice
# - return_ldl is an optional logical flag.  If true, the lifting deposition
#   level (LDL) is returned instead of the LCL. 
# - return_min_lcl_ldl is an optional logical flag.  If true, the minimum of the
#   LCL and LDL is returned.

def lcl(p,T,rh=None,rhl=None,rhs=None,return_ldl=False,return_min_lcl_ldl=False):

   import math
   import scipy.special

   # Parameters
   Ttrip = 273.16     # K
   ptrip = 611.65     # Pa
   E0v   = 2.3740e6   # J/kg
   E0s   = 0.3337e6   # J/kg
   ggr   = 9.81       # m/s^2
   rgasa = 287.04     # J/kg/K 
   rgasv = 461        # J/kg/K 
   cva   = 719        # J/kg/K
   cvv   = 1418       # J/kg/K 
   cvl   = 4119       # J/kg/K 
   cvs   = 1861       # J/kg/K 
   cpa   = cva + rgasa
   cpv   = cvv + rgasv

   # The saturation vapor pressure over liquid water
   def pvstarl(T):
      return ptrip * (T/Ttrip)**((cpv-cvl)/rgasv) * \
         math.exp( (E0v - (cvv-cvl)*Ttrip) / rgasv * (1/Ttrip - 1/T) )
   
   # The saturation vapor pressure over solid ice
   def pvstars(T):
      return ptrip * (T/Ttrip)**((cpv-cvs)/rgasv) * \
         math.exp( (E0v + E0s - (cvv-cvs)*Ttrip) / rgasv * (1/Ttrip - 1/T) )

   # Calculate pv from rh, rhl, or rhs
   rh_counter = 0
   if rh  is not None:
      rh_counter = rh_counter + 1
   if rhl is not None:
      rh_counter = rh_counter + 1
   if rhs is not None:
      rh_counter = rh_counter + 1
   if rh_counter != 1:
      print(rh_counter)
      exit('Error in lcl: Exactly one of rh, rhl, and rhs must be specified')
   if rh is not None:
      # The variable rh is assumed to be 
      # with respect to liquid if T > Ttrip and 
      # with respect to solid if T < Ttrip
      if T > Ttrip:
         pv = rh * pvstarl(T)
      else:
         pv = rh * pvstars(T)
      rhl = pv / pvstarl(T)
      rhs = pv / pvstars(T)
   elif rhl is not None:
      pv = rhl * pvstarl(T)
      rhs = pv / pvstars(T)
      if T > Ttrip:
         rh = rhl
      else:
         rh = rhs
   elif rhs is not None:
      pv = rhs * pvstars(T)
      rhl = pv / pvstarl(T)
      if T > Ttrip:
         rh = rhl
      else:
         rh = rhs
   if pv > p:
       exit('pv > p')

   # Calculate lcl_liquid and lcl_solid
   qv = rgasa*pv / (rgasv*p + (rgasa-rgasv)*pv)
   rgasm = (1-qv)*rgasa + qv*rgasv
   cpm = (1-qv)*cpa + qv*cpv
   if rh == 0:
      return cpm*T/ggr
   aL = -(cpv-cvl)/rgasv + cpm/rgasm
   bL = -(E0v-(cvv-cvl)*Ttrip)/(rgasv*T)
   cL = pv/pvstarl(T)*math.exp(-(E0v-(cvv-cvl)*Ttrip)/(rgasv*T))
   aS = -(cpv-cvs)/rgasv + cpm/rgasm
   bS = -(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T)
   cS = pv/pvstars(T)*math.exp(-(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T))
   lcl = cpm*T/ggr*( 1 - \
      bL/(aL*scipy.special.lambertw(bL/aL*cL**(1/aL),-1).real) )
   ldl = cpm*T/ggr*( 1 - \
      bS/(aS*scipy.special.lambertw(bS/aS*cS**(1/aS),-1).real) )

   # Return either lcl or ldl
   if return_ldl and return_min_lcl_ldl:
      exit('return_ldl and return_min_lcl_ldl cannot both be true')
   elif return_ldl:
      return ldl
   elif return_min_lcl_ldl:
      return min(lcl,ldl)
   else:
      return lcl

# from math import log10
from numpy import log10

def RH_water_ice(rv,tt,P):
    Patm = 101325 # Pa
    mw=18.0160 # molecular weight of water
    md=28.9660 # molecular weight of dry air
    fact=rv*md/mw
    tt0=273.16; a=-2663.5 ; b=12.537 # constants
    ewice=10**(a/tt+b)
    ewliq=Patm*10**(10.79586*(1-tt0/tt)-5.02808*log10(tt/tt0) + 1.50474*1e-4*(1-10**(-8.29692*(tt/tt0-1))) + 0.42873*1e-3*(10**(4.76955*(1-tt0/tt))-1)-2.2195983)    
    rhl =  P/ewliq*fact/(1+fact)*100.
    rhs =  P/ewice*fact/(1+fact)*100.
    return rhl,rhs

from numpy import log,maximum
def dewpoint(T,RH):
    T0=273.16
    b = 17.67
    c = 243.5
    gamma = log(maximum(RH,0.01)/100) + b*(T-T0)/(c+(T-T0))
    return c*gamma / (b-gamma) + T0

#%% Functions without ZS
@njit()
def horizontal_convergence(U,V,dx):
    CONV = np.zeros_like(U)
    CONV[:,:-1,:] -= ( V[:,1:,:] - V[:,:-1,:] )/dx
    CONV[:,-1,:] -= ( V[:,0,:] - V[:,-1,:] )/dx
    CONV[:,:,:-1] -= ( U[:,:,1:] - U[:,:,:-1] )/dx
    CONV[:,:,-1] -= ( U[:,:,0] - U[:,:,-1] )/dx
    return CONV

@njit(parallel=True)
def RG_lin_interp(VAR,Z,Zm): # Regular Grid linear interpolation
    # VAR (3D) =  VARiable defined on Zm altitude levels
    # Z (1D) =  altitude levels on which new_VAR in interpolated
    # Zm (1D) = model altitude levels
    nzVAR,nx,ny = np.shape(VAR)
    nz = np.shape(Z)
    new_VAR = np.zeros((nz,nx,ny),dtype=VAR.dtype)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                l = 0
                while Zm[l+1]<Z[k]:
                    l+=1
                if l<nzVAR-1:
                    dZ = Zm[l+1]-Zm[l]
                    new_VAR[k,i,j] = ( VAR[l,i,j]*(Zm[l+1]-Z[k]) + VAR[l+1,i,j]*(Z[k]-Zm[l]) )/dZ
    return new_VAR

#%% Functions with ZS

@njit(parallel=True)
def AL_lin_interp(VAR,AL,Zm,ZS): # Altitude Levels linear interpolation
    # VAR (3D) =  VARiable defined on Zm levels with Gal-Chen and Somerville terrain-following coordinates
    # AL (1D) =  altitude levels on which VAL in interpolated
    # Zm (1D) = terrain-following model levels
    # ZS (2D) = surface altitude
    nzVAR,ny,nx = np.shape(VAR)
    nz, = np.shape(AL)
    ZTOP = (Zm[-1]+Zm[-2])/2
    # ZTOP = Zm[-1]
    VAL = np.full((nz,ny,nx), np.nan,dtype=VAR.dtype)
    for j in prange(ny):
        for i in range(nx):
            zs = ZS[j,i]
            l = 0
            for k in range(nz):
                if AL[k]>zs:
                    zm = ZTOP * (AL[k]-zs) / (ZTOP-zs)
                    while Zm[l+1]<zm:
                        l+=1
                    if l<nzVAR-1:
                        dZ = Zm[l+1]-Zm[l]
                        VAL[k,j,i] = ( VAR[l,j,i]*(Zm[l+1]-zm) + VAR[l+1,j,i]*(zm-Zm[l]) )/dZ
    return VAL

@njit(parallel=True)
def AL_objects(VAR,AL,Zm,ZS): # Altitude Levels linear interpolation
    # VAR (3D) =  VARiable defined on Zm levels with Gal-Chen and Somerville terrain-following coordinates
    # AL (1D) =  altitude levels on which VAL in interpolated
    # Zm (1D) = terrain-following model levels
    # ZS (2D) = surface altitude
    nzVAR,ny,nx = np.shape(VAR)
    nz, = np.shape(AL)
    ZTOP = (Zm[-1]+Zm[-2])/2
    VAL = -np.ones((nz,ny,nx),dtype=VAR.dtype)
    for j in prange(ny):
        for i in range(nx):
            zs = ZS[j,i]
            l = 0
            for k in range(nz):
                if AL[k]>zs:
                    zm = ZTOP * (AL[k]-zs) / (ZTOP-zs)
                    while Zm[l+1]<zm:
                        l+=1
                    if l<nzVAR-1:
                        if (Zm[l+1]-zm)>(zm-Zm[l]):
                            VAL[k,j,i] = VAR[l,j,i] 
                        else:
                            VAL[k,j,i] = VAR[l+1,j,i]
    return VAL

@njit(parallel=True)
def AGL_lin_interp(VAR,AGL,Zm,ZS): # Above Ground Level linear interpolation
    # VAR (3D) =  VARiable defined on Zm levels with Gal-Chen and Somerville terrain-following coordinates
    # AGL =  above ground level on which VAGL in interpolated
    # Zm (1D) = terrain-following model levels
    # ZS (2D) = surface altitude
    nzVAR,ny,nx = np.shape(VAR)
    ZTOP = (Zm[-1]+Zm[-2])/2
    VAGL = np.zeros((ny,nx), dtype=VAR.dtype)
    for j in prange(ny):
        for i in range(nx):
            zm = ZTOP * AGL / (ZTOP-ZS[j,i])
            l = 0
            while Zm[l+1]<zm:
                l+=1
            if l<nzVAR-1:
                dZ = Zm[l+1]-Zm[l]
                VAGL[j,i] = ( VAR[l,j,i]*(Zm[l+1]-zm) + VAR[l+1,j,i]*(zm-Zm[l]) )/dZ
    return VAGL

@njit(parallel=True)
def AGL_lin_anomaly(VAGL,AGL,z,mean,ZS): 
    # VAGL (2D) is obtained with AGL_lin_interp
    # mean (1D) is nanmean(axis=(1,2)) of VAL obtained with AL_lin_interp
    VAGLA = np.copy(VAGL)
    ny,nx = np.shape(VAGLA)
    for j in prange(ny):
        for i in range(nx):
            zVAGL = ZS[j,i] + AGL
            k = 0
            while zVAGL>z[k+1] or np.isnan(mean[k]):
                k+=1
            dz = z[k+1]-z[k]
            VAGLA[j,i] -= ( mean[k]*(z[k+1]-zVAGL) + mean[k+1]*(zVAGL-z[k]) )/dz
    return VAGLA

@njit(parallel=True)
def MO_flux(MO,U,V,RHO,RV,RT,TH):
    nz,ny,nx = np.shape(U)
    flux =  np.zeros((4,nz),dtype=U.dtype)
    for j in range(ny):
        for i in range(nx):
            i0 = i-1 if i>=1 else nx-1
            j0 = j-1 if j>=1 else ny-1
            for j_,i_,wind in [ (j,i0,U[:,j,i]) , (j0,i,V[:,j,i]) ]:
                if MO[j_,i_] != MO[j,i]:
                    sign = 1. if MO[j,i] else -1.
                    for k in range(nz):
                        massflux = sign*wind[k]*(RHO[k,j_,i_]+RHO[k,j,i])/2
                        flux[0,k] += massflux
                        flux[1,k] += massflux*(RV[k,j_,i_]+RV[k,j,i])/2 # vapor flux
                        flux[2,k] += massflux*(RT[k,j_,i_]+RT[k,j,i])/2 # water flux
                        flux[3,k] += massflux*(TH[k,j_,i_]+TH[k,j,i])/2 # heat flux
    return flux

@njit(parallel=True)
def boundary_layer_diags(Zm,ZS,rc,thv,rho,W,U,V,thermals,deltaTH=0.3):
    nz,ny,nx = np.shape(rc)
    BLH = np.full((ny,nx),np.nan,dtype=floatype) # Boundary Layer Height
    THVBL = np.zeros((ny,nx),dtype=floatype) 
    WBL = np.zeros((ny,nx),dtype=floatype)
    UBL = np.zeros((ny,nx),dtype=floatype)
    VBL = np.zeros((ny,nx),dtype=floatype)
    UBLH = np.zeros((ny,nx),dtype=floatype)
    VBLH = np.zeros((ny,nx),dtype=floatype)
    CBM = np.zeros((ny,nx),dtype=intype) # Cumulus Base Mask
    TTM = np.zeros((ny,nx),dtype=intype) # Thermal Top Mask
    ZTOP = (Zm[-1]+Zm[-2])/2
    for j in prange(ny):
        for i in range(nx):
            mean_thv = thv[0,j,i]
            mean_w = W[0,j,i]
            mean_u = U[0,j,i]
            mean_v = V[0,j,i]
            mass = Zm[0]*rho[0,j,i]
            insideBL = True
            for h in range(0,nz-1):
                if rc[h,j,i]>1e-6:
                    if insideBL == True:
                        BLH[j,i] = ZS[j,i] + (ZTOP-ZS[j,i]) * Zm[h]/ZTOP
                        CBM[j,i] = 1
                        if thermals[h-1,j,i] > 0: TTM[j,i] = 1
                        THVBL[j,i] = mean_thv
                        WBL[j,i] = mean_w
                        UBL[j,i] = mean_u
                        VBL[j,i] = mean_v
                    break
                if thv[h,j,i] > mean_thv + deltaTH:
                    if insideBL == True:
                        insideBL = False
                        BLH[j,i] = ZS[j,i] + (ZTOP-ZS[j,i]) * Zm[h]/ZTOP
                        if thermals[h-1,j,i] > 0: TTM[j,i] = 1
                        THVBL[j,i] = mean_thv
                        WBL[j,i] = mean_w
                        UBL[j,i] = mean_u
                        VBL[j,i] = mean_v
                elif insideBL == False:
                    insideBL = True
                layer_mass = (rho[h+1,j,i]+rho[h,j,i])/2 * (Zm[h+1]-Zm[h])
                mean_thv = ( mean_thv*mass + (thv[h+1,j,i]+thv[h,j,i])/2 * layer_mass ) / ( mass+layer_mass)
                mean_w = ( mean_w*mass + W[h+1,j,i] * layer_mass ) / ( mass+layer_mass)
                mean_u = ( mean_u*mass + U[h+1,j,i] * layer_mass ) / ( mass+layer_mass)
                mean_v = ( mean_v*mass + V[h+1,j,i] * layer_mass ) / ( mass+layer_mass)
                mass += layer_mass
            
            UBLH[j,i] = U[h,j,i]
            VBLH[j,i] = V[h,j,i]
    return BLH,CBM,TTM,THVBL,WBL,UBL,VBL,UBLH,VBLH

@njit(parallel=True)
def surface_layer_diags(Zm,ZS,thv,rho,W,U,V,thermals,maxD=100.):
    nz,ny,nx = np.shape(thv)
    SLD = np.full((ny,nx),np.nan,dtype=floatype) # Surface Layer Depth
    THVSL = np.zeros((ny,nx),dtype=floatype) # mean THeta_V of Surface Layer
    WSL = np.zeros((ny,nx),dtype=floatype) # mean W of Surface Layer
    USL = np.zeros((ny,nx),dtype=floatype)
    VSL = np.zeros((ny,nx),dtype=floatype)
    # MFSL = np.zeros((ny,nx),dtype=floatype)# Mass Flux of Surface Layer
    TBM = np.zeros((ny,nx),dtype=intype)# Thermal Base Mask
    ZTOP = (Zm[-1]+Zm[-2])/2
    for j in prange(ny):
        for i in range(nx):
            mean_thv = thv[0,j,i]
            mean_w = W[0,j,i]
            mean_u = U[0,j,i]
            mean_v = V[0,j,i]
            mass = Zm[0]*rho[0,j,i]
            dthvdZm0 = (thv[1,j,i]-thv[0,j,i])/(Zm[1]-Zm[0])
            for h in range(0,nz-1):
                depth = (ZTOP-ZS[j,i]) * Zm[h]/ZTOP
                if depth > maxD or (thv[h+1,j,i]-thv[h,j,i])/(Zm[h+1]-Zm[h])/dthvdZm0 < 0.1 : # 0.2 comes from Prantdl slope wind theory, it is where d(n)=0
                    SLD[j,i] = min( maxD, (ZTOP-ZS[j,i]) * Zm[h]/ZTOP)
                    if thermals[h,j,i] > 0: TBM[j,i] = 1
                    break
                else:
                    layer_mass = (rho[h+1,j,i]+rho[h,j,i])/2 * (Zm[h+1]-Zm[h])
                    mean_thv = ( mean_thv*mass + (thv[h+1,j,i]+thv[h,j,i])/2 * layer_mass ) / ( mass+layer_mass)
                    mean_w = ( mean_w*mass + W[h+1,j,i] * layer_mass ) / ( mass+layer_mass)
                    mean_u = ( mean_u*mass + U[h+1,j,i] * layer_mass ) / ( mass+layer_mass)
                    mean_v = ( mean_v*mass + V[h+1,j,i] * layer_mass ) / ( mass+layer_mass)
                    mass += layer_mass
            THVSL[j,i] = mean_thv
            WSL[j,i] = mean_w
            USL[j,i] = mean_u
            VSL[j,i] = mean_v
            # MFSL[j,i] = mean_w*mass/SLD[j,i]
    return SLD,TBM,THVSL,WSL,USL,VSL

@njit(parallel=True)
def AL_SL(AL,ZS,D): # Altitude Levels surface layer definition
    # AL (1D) =  altitude levels on which VAL in interpolated
    # ZS (2D) = surface altitude
    # D (2D) = depth of the surface layer
    ny,nx = np.shape(ZS)
    nz, = np.shape(AL)
    SL = np.zeros((nz,ny,nx),dtype=np.bool_)
    for j in prange(ny):
        for i in range(nx):
            zs = ZS[j,i]
            for k in range(nz):
                if AL[k]>zs and AL[k]<zs+D[j,i]:
                    SL[k,j,i] = True
    return SL
