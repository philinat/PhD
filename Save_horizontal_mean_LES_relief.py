#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:13:05 2023

@author: philippotn
"""

import time
import numpy as np
import xarray as xr
from numba import njit,prange
import os
import warnings
warnings.filterwarnings('ignore')
floatype = np.float32

# simu = 'AMOPL'
# simu = 'A0W0V'
# simu = 'ATRI2'
# simu = 'ACON2'
# simu = 'AMO20'
# simu = 'CTRI2'
simu = 'CFLAT'
# simu = 'CMO10'

orog = False

userPath = '/cnrm/tropics/user/philippotn'
# userPath = '/home/philippotn'

if userPath == '/home/philippotn':
    dataPath = userPath+'/Documents/NO_SAVE/'
else:
    dataPath = userPath+'/LES_'+simu+'/NO_SAVE/'

savePath = dataPath
if not os.path.exists(savePath): os.makedirs(savePath) ; print('savePath directory created !')

if simu == 'AMOPL':
    seg = 'M200m' ; lFiles = [dataPath + 'AMOPL.1.200m1.OUT.{:03d}.nc'.format(i) for i in range(1,241,1)] + [dataPath + 'AMOPL.1.200m2.OUT.{:03d}.nc'.format(i) for i in range(1,722,1)]
elif simu== 'A0W0V':
    seg = 'S200m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,962,1)]
elif simu== 'ATRI2':
    seg = 'S200m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,962,1)]
elif simu== 'ACON2':
    seg = 'S200m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,962,1)]
elif simu== 'AMO20':
    seg = 'M200m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,961,1)]
elif simu== 'CTRI2':
    seg = 'M100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,872,1)]
elif simu== 'CFLAT':
    seg = 'S100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,872,1)]
elif simu== 'CMO10':
    seg = 'M100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(2,3,1)]

f0 = xr.open_dataset(lFiles[0])

nx1 = 1 ; nx2 = len(f0.ni)-1
ny1 = 1 ; ny2 = len(f0.nj)-1
nz1 = 1 ; nz2 = len(f0.level)-1

x = np.array(f0.ni)[nx1:nx2]
y = np.array(f0.nj)[ny1:ny2]
z = np.array(f0.level)[nz1:nz2]
Zm = np.array(f0.level)[nz1:]

dx = x[1]-x[0]
nt,nz,ny,nx = len(lFiles),len(z),len(y),len(x)

if orog:
    ZS = xr.open_dataset(dataPath+simu+'_init_'+seg+'_pgd.nc')['ZS'][ny1:ny2,nx1:nx2].data
    MO = ZS>0. # Mountain region (upper one)
    PL = np.logical_not(MO) # Plain region (lower one)
else:
    ZS = np.zeros((ny,nx))
    
Z = np.copy(z)

#%%
@njit(parallel=True)
def interp_on_altitude_levels(var,Z,Zm,ZS): # var has to be np.float32 ???
    # var (3D) =  variable defined on Zm levels with Gal-Chen and Somerville terrain-following coordinates
    # Z (1D) =  altitude levels on which new_var in interpolated
    # Zm (1D) = terrain-following model levels
    # ZS (2D) = surface altitude
    _,ny,nx = np.shape(var)
    nz, = np.shape(Z)
    ZTOP = Zm[-1]
    new_var = np.full((nz,ny,nx), np.nan,dtype=var.dtype)
    for j in prange(ny):
        for i in range(nx):
            for k in range(nz):
                zs = ZS[j,i]
                l = 0
                if Z[k]<zs:
                    continue
                else:
                    zm = ZTOP * (Z[k]-zs) / (ZTOP-zs)
                    while Zm[l+1]<zm:
                        l+=1
                    dZ = Zm[l+1]-Zm[l]
                    new_var[k,j,i] = ( var[l,j,i]*(Zm[l+1]-zm) + var[l+1,j,i]*(zm-Zm[l]) )/dZ
    return new_var
#%%
nvar = 7
# timeH = np.zeros(nt)
data_mean = np.empty((nvar,nt,nz),dtype=floatype)
precip_mean = np.zeros(nt,dtype=floatype)
if orog:
    data_mean_MO = np.empty((nvar,nt,nz),dtype=floatype)
    data_mean_PL = np.empty((nvar,nt,nz),dtype=floatype)
    precip_mean_MO = np.zeros(nt,dtype=floatype)
    precip_mean_PL = np.zeros(nt,dtype=floatype)

times = f0.time.data
for it,fname in enumerate(lFiles):
    time0 = time.time()
    f = xr.open_dataset(fname)
    def npf(v,dtype=floatype):
        return np.array(f[v][0,nz1:nz2,ny1:ny2,nx1:nx2],dtype=dtype)
    if it>0: times = np.concatenate((times,f.time.data))
    timeH = ' {:02d}h{:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute)) # year = str(f.time.data[0])[:10]+
    # timeH[it] = f.time.dt.hour + f.time.dt.minute/60
    print(time.ctime()[-13:-5], timeH ,end=' ')
    
    TH = npf('THT')
    RV = npf('RVT')
    RC = npf('RCT')+npf('RIT')
    RP = npf('RRT')+npf('RST')+npf('RGT')
    P = npf('PABST')
    W = npf('WT')
    TR = npf('SVT001')
    precip = np.array(f.ACPRRSTEP[0,ny1:ny2,nx1:nx2],dtype=floatype)
    
    print(' Read :',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
    # T = TH * (P/100000)**(2/7)
    THV = TH * (1.+ 1.61*RV)/(1.+RV+RC+RP)
    
    if orog:
        TH = interp_on_altitude_levels( TH,Z,Zm,ZS)
        RV = interp_on_altitude_levels(RV,Z,Zm,ZS)
        THV = interp_on_altitude_levels( THV,Z,Zm,ZS)
        RC = interp_on_altitude_levels(RC,Z,Zm,ZS)
        P = interp_on_altitude_levels( P,Z,Zm,ZS)
        TR = interp_on_altitude_levels(TR,Z,Zm,ZS)
        # W = interp_on_altitude_levels(TR,Z,Zm,ZS)
        
    precip_mean[it] = np.mean(precip)
    data_mean[0,it] = np.nanmean(TH,axis=(1,2)) # K
    data_mean[1,it] = np.nanmean(RV,axis=(1,2)) # g/kg
    data_mean[2,it] = np.nanmean(THV,axis=(1,2)) # K
    data_mean[3,it] = np.nanmean( RC>1e-6 ,axis=(1,2)) # cloud fraction
    data_mean[4,it] = np.nanmean(P,axis=(1,2)) # Pa
    data_mean[5,it] = np.nanmean(TR,axis=(1,2))
    data_mean[6,it] = np.nanmean( interp_on_altitude_levels(W,Z,Zm,ZS) ,axis=(1,2)) # m/s
    
    if orog:
        precip_mean_MO[it] = np.mean(precip[MO])
        data_mean_MO[0,it] = np.nanmean(TH[:,MO],axis=1) # K
        data_mean_MO[1,it] = np.nanmean(RV[:,MO],axis=1) # g/kg
        data_mean_MO[2,it] = np.nanmean(THV[:,MO],axis=1) # K
        data_mean_MO[3,it] = np.nanmean( RC[:,MO]>1e-6,axis=1) # cloud fraction
        data_mean_MO[4,it] = np.nanmean(P[:,MO],axis=1) # Pa
        data_mean_MO[5,it] = np.nanmean(TR[:,MO],axis=1)
        data_mean_MO[6,it] = np.nanmean(W[:,MO],axis=1) # m/s
        
        precip_mean_PL[it] = np.mean(precip[PL])
        data_mean_PL[0,it] = np.nanmean(TH[:,PL],axis=1) # K
        data_mean_PL[1,it] = np.nanmean(RV[:,PL],axis=1) # g/kg
        data_mean_PL[2,it] = np.nanmean(THV[:,PL],axis=1) # K
        data_mean_PL[3,it] = np.nanmean( RC[:,PL]>1e-6,axis=1) # cloud fraction
        data_mean_PL[4,it] = np.nanmean(P[:,PL],axis=1) # Pa
        data_mean_PL[5,it] = np.nanmean(TR[:,PL],axis=1)
        data_mean_PL[6,it] = np.nanmean(W[:,PL],axis=1) # m/s
    
    print('Compute :' ,str(round(time.time()-time0,1))+' s')

#%%
if orog:
    ds = xr.Dataset(
        {
        "ZS": (("y", "x"), ZS),
        "TH": (("time", "z"), data_mean[0]),
        "RV": (("time", "z"), data_mean[1]),
        "THV": (("time", "z"), data_mean[2]),
        "CF": (("time", "z"), data_mean[3]),
        "P": (("time", "z"), data_mean[4]),
        "TR": (("time", "z"), data_mean[5]),
        "W": (("time", "z"), data_mean[6]),
        "PR": (("time",),precip_mean),
        
        "MO": (("y", "x"), MO),
        "TH_MO": (("time", "z"), data_mean_MO[0]),
        "RV_MO": (("time", "z"), data_mean_MO[1]),
        "THV_MO": (("time", "z"), data_mean_MO[2]),
        "CF_MO": (("time", "z"), data_mean_MO[3]),
        "P_MO": (("time", "z"), data_mean_MO[4]),
        "TR_MO": (("time", "z"), data_mean_MO[5]),
        "W_MO": (("time", "z"), data_mean_MO[6]),
        "PR_MO": (("time",),precip_mean_MO),
        
        "PL": (("y", "x"), PL),
        "TH_PL": (("time", "z"), data_mean_PL[0]),
        "RV_PL": (("time", "z"), data_mean_PL[1]),
        "THV_PL": (("time", "z"), data_mean_PL[2]),
        "CF_PL": (("time", "z"), data_mean_PL[3]),
        "P_PL": (("time", "z"), data_mean_PL[4]),
        "TR_PL": (("time", "z"), data_mean_PL[5]),
        "W_PL": (("time", "z"), data_mean_PL[6]),
        "PR_PL": (("time",),precip_mean_PL)
        },
        coords={"time": times,
            "z": Z,
            "y": y,
            "x": x},
    )
else:
    ds = xr.Dataset(
        {
        "ZS": (("y", "x"), ZS),
        "TH": (("time", "z"), data_mean[0]),
        "RV": (("time", "z"), data_mean[1]),
        "THV": (("time", "z"), data_mean[2]),
        "CF": (("time", "z"), data_mean[3]),
        "P": (("time", "z"), data_mean[4]),
        "TR": (("time", "z"), data_mean[5]),
        "W": (("time", "z"), data_mean[6]),
        "PR": (("time",),precip_mean)
        },
        coords={"time": times,
            "z": Z,
            "y": y,
            "x": x},
)

ds.to_netcdf(savePath+simu+'_'+seg+'_mean.nc', encoding={"time": {'units': 'days since 1900-01-01'}})

# np.savez_compressed(savePath+simu+'_'+seg+'_mean',TH=data_mean[0],RV=data_mean[1],CF=data_mean[2],TR=data_mean[3],PR=precip_mean,Z=Z,T=timeH,ZS=ZS)

#%%
# import matplotlib.pyplot as plt
# fig,axs = plt.subplots(ncols=5)
# for it,tt in enumerate(times):
#     axs[0].plot(data_mean[0,it],Z,label=tt)
#     axs[1].plot(data_mean[1,it],Z,label=tt)
#     axs[2].plot(data_mean[2,it],Z,label=tt)
#     axs[3].plot(data_mean[3,it],Z,label=tt)
# axs[4].plot(timeH,precip_mean)
# axs[0].legend()
