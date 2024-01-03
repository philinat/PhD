#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:24:55 2023

@author: philippotn
"""
import sys
import numpy as np
import os
from numba import njit,prange
import xarray as xr

# - Data LES
# simu = 'AMOPL'
# simu = 'ATRI2'
simu = 'AMO20'

userPath = '/cnrm/tropics/user/philippotn'
# userPath = '/home/philippotn'

if userPath == '/home/philippotn':
    dataPath = userPath+'/Documents/SIMU_LES/'
    sys.path.append(userPath+'/Documents/Codes_python/objects-master/src/')
    
else:
    dataPath = userPath+'/LES_'+simu+'/NO_SAVE/'
    sys.path.append(userPath+'/Code_LES/objects-master/src/')

from identification_methods import identify


if simu == 'AMOPL':
    lFiles = [dataPath + 'AMOPL.1.200m2.OUT.{:03d}.nc'.format(i) for i in [1,181,361,541,721]]
elif simu== 'ATRI2':
    lFiles = [dataPath + simu+'.1.S200m.OUT.{:03d}.nc'.format(i) for i in range(1,962,60)]
    
f0 = xr.open_dataset(lFiles[0])
z = np.array(f0.level)
#%% --------- identify -----
listVarNames = ['SVT001','THT','RVT','RCT','RIT','PABST']

@njit(parallel=True)
def core_local_minimum_of_tracer_or_cloud_base(z,rc,thv,rho,tracer):
    nz,ny,nx = np.shape(tracer)
    mask = np.zeros((1,nz,ny,nx),dtype='bool')
    for j in prange(1,ny-1):
        for i in range(1,nx-1):
            mean_thv = thv[1,j,i]
            mass = z[1]*rho[1,j,i]
            h=2
            while h < len(z)-1:
                if rc[h,j,i]>1e-6:
                    mask[0,h,j,i] = True
                    break
                if thv[h,j,i] > mean_thv + 0.3:
                    break
                else:
                    layer_mass = (rho[h+1,j,i]+rho[h,j,i])/2 * (z[h+1]-z[h])
                    mean_thv = ( mean_thv*mass + (thv[h+1,j,i]+thv[h,j,i])/2 * layer_mass ) / ( mass+layer_mass)
                    mass += layer_mass
                h+=1
                
            while h < len(z)-1:
                tr = tracer[h,j,i]
                if rc[h,j,i]>1e-6 and tr>tracer[h-1,j,i] and tr>tracer[h+1,j,i] and tr>tracer[h,j+1,i] and tr>tracer[h,j-1,i] and tr>tracer[h,j,i+1] and tr>tracer[h,j,i-1]:
                    mask[0,h,j,i] = True
                h+=1
    return mask


def cloudMask(dictVars) : 
    rc = dictVars['RCT']
    ri = dictVars['RIT']
    mask = np.zeros_like(rc)
    mask[(rc+ri)>10**(-6)] = 1
    return mask

def coreMask(dictVars) :
    tracer = dictVars['SVT001']
    rc = dictVars['RCT']
    ri = dictVars['RIT']
    rcloud = rc+ri
    th = dictVars['THT']
    rv = dictVars['RVT']
    P = dictVars['PABST']
    tracer = dictVars['SVT001']
    thv = th * (1 + 0.61*rv)
    rho = P/287.04/ (thv * (P/100000)**(2/7))
    return -tracer , core_local_minimum_of_tracer_or_cloud_base(z,rcloud[0],thv[0],rho[0],tracer[0])

# def thermalMask(dictVars):
#     TR = dictVars['SVT001']
#     W = dictVars['WT']
#     TR_std = np.std(TR,axis=(1,2))
#     TR_mean = np.mean(TR,axis=(1,2))
#     TR_stdmin = 0.05/(z+dz/2) * np.cumsum(dz*TR_std)
#     TR_threshold = TR_mean+np.maximum(TR_std,TR_stdmin)
#     return np.logical_and( TR>TR_threshold , W>0. )
    
for f in lFiles[:]:
    print("\n "+f)
    objects, tmp = identify(f, listVarNames, cloudMask, funcWatershed=coreMask, name="tracer_clouds",overwrite=True,delete=0,deleteCores=0)