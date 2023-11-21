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
floatype = np.float32

simu = 'AMOPL'
orog = True

userPath = '/cnrm/tropics/user/philippotn'
# userPath = '/home/philippotn'

if userPath == '/home/philippotn':
    dataPath = userPath+'/Documents/SIMU_LES/'
else:
    dataPath = userPath+'/LES_'+simu+'/SIMU_LES/'

savePath = dataPath

if simu == 'AMOPL':
    # lFiles = [dataPath + 'AMOPL.1.200m1.OUT.{:03d}.nc'.format(i) for i in [60, 240]]+[dataPath + 'AMOPL.1.200m2.OUT.{:03d}.nc'.format(i) for i in [180,360,540,720]]
    lFiles = [dataPath + 'AMOPL.1.200m1.OUT.{:03d}.nc'.format(i) for i in range(1,241,1)] + [dataPath + 'AMOPL.1.200m2.OUT.{:03d}.nc'.format(i) for i in range(1,722,1)]
    
if not os.path.exists(savePath): os.makedirs(savePath) ; print('Directory created !')
f0 = xr.open_dataset(lFiles[0])

nx1 = 1 ; nx2 = len(f0.ni)-1
ny1 = 1 ; ny2 = len(f0.nj)-1
nz1 = 1 ; nz2 = len(f0.level)-1

x = np.array(f0.ni)[nx1:nx2]
y = np.array(f0.nj)[ny1:ny2]
z = np.array(f0.level)[nz1:nz2]
Zm = np.array(f0.level)[nz1:]

dx = x[1]-x[0]
nt,nz,nx,ny = len(lFiles),len(z),len(x),len(y)

if orog:
    ZS = xr.open_dataset(dataPath+simu+'_init_R'+str(round(dx))+'m_pgd.nc')['ZS'][ny1:ny2,nx1:nx2].data
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
    _,nx,ny = np.shape(var)
    nz, = np.shape(Z)
    ZTOP = Zm[-1]
    new_var = np.full((nz,nx,ny), np.nan,dtype=var.dtype)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                zs = ZS[i,j]
                l = 0
                if Z[k]<zs:
                    continue
                else:
                    zm = ZTOP * (Z[k]-zs) / (ZTOP-zs)
                    while Zm[l+1]<zm:
                        l+=1
                    dZ = Zm[l+1]-Zm[l]
                    new_var[k,i,j] = ( var[l,i,j]*(Zm[l+1]-zm) + var[l+1,i,j]*(zm-Zm[l]) )/dZ
    return new_var

    
#%%

def npf(v,dtype=floatype):
    return np.array(f[v][0,nz1:nz2,ny1:ny2,nx1:nx2],dtype=dtype)
def npf_interp(v,dtype=floatype):
    return np.array( interp_on_altitude_levels(f[v][0,nz1:nz2,ny1:ny2,nx1:nx2].data,Z,Zm,ZS) ,dtype=dtype)

data_mean = np.empty((4,nt,nz),dtype=floatype)
precip_mean = np.zeros(nt)
timeH = np.zeros(nt)
times = []
for it,fname in enumerate(lFiles):
    time0 = time.time()
    f = xr.open_dataset(fname)
    times.append(' {:02d}h{:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute))) # year = str(f.time.data[0])[:10]+
    timeH[it] = f.time.dt.hour + f.time.dt.minute/60
    print(times[-1],end=' ')
    
    # RV = npf('RVT',dtype=np.float32)
    
    # RP = npf('RRT',dtype=np.float32)+npf('RST',dtype=np.float32)+npf('RGT',dtype=np.float32)
    # P = npf('PABST',dtype=np.float32)
    # TH = npf('THT',dtype=np.float32)
    
    # T = TH * (P/100000)**(2/7)
    # THV = TH * (1.+ 1.61*RV)/(1.+RV+RC+RP)
    # rhl,rhi = RH_water_ice(RV,T,P)
    # RH = np.maximum(rhl,rhi)
    
    # Pz = interp_on_altitude_levels( P,Z,Zm,ZS)
    # THVz = interp_on_altitude_levels( THV,Z,Zm,ZS)
    
    precip_mean[it] = np.mean(f.ACPRRSTEP[0,ny1:ny2,nx1:nx2].data)
    
    data_mean[0,it] = np.nanmean(npf_interp('THT'),axis=(1,2))
    data_mean[1,it] = np.nanmean(npf_interp('RVT'),axis=(1,2))*1000
    RCZ = interp_on_altitude_levels(npf('RCT')+npf('RIT'),Z,Zm,ZS)
    data_mean[2,it] = np.nanmean( RCZ>1e-6 ,axis=(1,2)) # cloud fraction
    data_mean[3,it] = np.nanmean(npf_interp('SVT001'),axis=(1,2))
    
    print(time.ctime()[-13:-5] ,str(round(time.time()-time0,1))+' s')
    
np.savez_compressed(savePath+simu+'_'+str(round(dx))+'_mean',TH=data_mean[0],RV=data_mean[1],CF=data_mean[2],TR=data_mean[3],PR=precip_mean,Z=Z,T=timeH,ZS=ZS)

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
