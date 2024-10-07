#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:58:35 2023

@author: philippotn
"""


import numpy as np
import xarray as xr
import sparse # pip install sparse
import math
import time
import os
from numba import njit,prange
import numba
numba.set_num_threads(numba.config.NUMBA_NUM_THREADS//2) # Limit CPU parallelization to half of the cores ( usefull if you want run other codes at the same time )

floatype = np.float32
idx_dtype = np.uint32
intype = np.int32

minimal_setting_for_time_tracking = True
saveFormat = 'npz' # 'npz' or 'sparse' ## npz seems as efficient as sparse, and more convenient
end = '--- '

simu = 'AROME'

# you need a tFile to get 
if simu == 'AROME':
    # file = '/cnrm/tropics/commun/DATACOMMUN/AROME_OM/ARCHIVE/R00/GNXO/charac/hmix_SMOC_atlant_freq1h_EUREC4A_1weeks_watershed_False_ANO_THH_0.3_pix_0.nc'
    # file = '/cnrm/tropics/commun/DATACOMMUN/AROME_OM/ARCHIVE/R00/GNXO/charac/LCC_atlant_freq1h_allperiod_EUREC4A_1weeks_watershed_False_LCC_100_pix_0.nc'
    # file = '/cnrm/tropics/commun/DATACOMMUN/AROME_OM/ARCHIVE/R00/GJXH/LCC_atlant_freq1h_allperiod_EUREC4A.nc'
    
    path = '/cnrm/tropics/user/philippotn/AROME/allperiod_U10_V10_LCC/'
    lFiles = [path + 'split_{:06d}.nc'.format(i) for i in range(1,745)]
it0 = 1 # Time index of the first file  
# savePath = '/cnrm/tropics/user/philippotn/'+simu+'/characs_objects/1H_coldpools/'
savePath = '/cnrm/tropics/user/philippotn/'+simu+'/characs_objects/1H_lowclouds/'

if not os.path.exists(savePath):
    os.makedirs(savePath)
    print('Directory created !')
    

# f = xr.open_dataset(file)
# nt = len(f.TIME)


#%% Main functions
@njit()
def charac_opti_2D(objects,variables):
    # charac = (mean,std,min,max) of 2D objects for each given 2D variables
    # area = area of objects
    # pos = (x,y,std) mean position of object, and its standard deviation 
    nx,ny = objects.shape
    nobj = np.max(objects)+1 # Objects needs to contains all the different labels from 0 (environment) to max(objects) # if not, use change_label(objects)
    charac = np.zeros((nobj,len(variables),4),dtype=floatype)
    area = np.zeros((nobj),dtype=intype)
    pos = np.zeros((nobj,3),dtype=floatype)
    
    for i in range(nx):
        for j in range(ny):
            obj = objects[i,j]
            area[obj] +=1
            
            if obj>0: # Compute mean position only for objects (not for environment)
                if area[obj]==1: 
                    pos[obj,0] = i
                    pos[obj,1] = j
                else:
                    dx = i-pos[obj,0]
                    if nx-abs(dx)<abs(dx): # Periodic distance
                        dx = -np.sign(dx)* (nx-abs(dx))
                    dy = j-pos[obj,1]
                    if ny-abs(dy)<abs(dy):
                        dy = -np.sign(dy)* (ny-abs(dy))
                    pos[obj,0] = (pos[obj,0] + dx/area[obj])%nx # Periodic position thank to %
                    pos[obj,1] = (pos[obj,1] + dy/area[obj])%ny
            
            for v,variable in enumerate(variables): # Compute incrementaly (mean,std,min,max)
                charac[obj,v,0] += variable[i,j]
                charac[obj,v,1] += variable[i,j]**2
                if charac[obj,v,2] > variable[i,j] or area[obj]==1:
                    charac[obj,v,2] = variable[i,j]
                if charac[obj,v,3] < variable[i,j] or area[obj]==1:
                    charac[obj,v,3] = variable[i,j]
                        
    for i in range(nx): # Loop to compute position variance from the previously computed mean position of objects
        for j in range(ny):
            obj = objects[i,j]
            if obj>0:
                dx = i-pos[obj,0]
                if nx-abs(dx)<abs(dx):
                    dx = -np.sign(dx)* (nx-abs(dx))
                dy = j-pos[obj,1]
                if ny-abs(dy)<abs(dy):
                    dy = -np.sign(dy)* (ny-abs(dy))
                pos[obj,2] += dx**2+dy**2
                        
    for obj in range(nobj): # Loop on objects
        if area[obj]>0:
            pos[obj,2] = np.sqrt(pos[obj,2]/area[obj])
            for v in range(len(variables)):
                charac[obj,v,0] /= area[obj] # Mean
                charac[obj,v,1] = np.sqrt( charac[obj,v,1]/area[obj] - charac[obj,v,0]**2 ) # Standard deviation as sqrt(E[X^2] - E[X]^2)
        else:
            charac[obj,:,:] = np.nan # Outside of objects charac and pos are undefined 
            pos[obj,:] = np.nan
    return charac,area,pos
                
@njit(parallel=True)
def change_label(objects): 
    # Function shifting the labels inside objects where there are holes (unused labels)
    # There are holes when identify is periodic
    # In the end : np.unique(objects) == np.arange(np.max(objects)+1)
    nx,ny = np.shape(objects)
    unique = np.unique(objects)
    old2new = np.zeros(unique[-1]+1,dtype=intype)
    for i,u in enumerate(unique):
        old2new[u] = i
        
    for i in prange(nx):
        for j in range(ny):
            obj = objects[i,j]
            if obj>0:
                objects[i,j] = old2new[obj]
            if obj<0:
                objects[i,j] = -1

#%% Main loop : prepare your 2D variables and characterize your objects

if minimal_setting_for_time_tracking:
    # for it in range(nt):
    for it,file in enumerate(lFiles[:]):
        time0 = time.time()
        # print(file)
        print('{:03d}'.format(it+1),end=' ')
        print("Reading file",end=end)
        f = xr.open_dataset(file)
        def npf(v,dtype=floatype, order='C'):
            return np.array(f[v][0,1:-1,1:-1],dtype=dtype, order=order)
        
        print("Objects",end=end)
        objects = npf('LCC_100',dtype=intype)
        change_label(objects)
        
        print("Variables",end=end)
        variables = np.array([npf('v10'),npf('u10')],dtype=floatype) # must contain the
        
        print("Charac",end=end)
        charac,area,pos = charac_opti_2D(objects,variables)
        
        if saveFormat == 'sparse':
            print("Sparse",end=end)
            sparse.save_npz(savePath+'charac.{:03d}'.format(it+it0), sparse.COO.from_numpy(charac,idx_dtype=idx_dtype,fill_value=np.nan) )
            sparse.save_npz(savePath+'area.{:03d}'.format(it+it0), sparse.COO.from_numpy(area,idx_dtype=idx_dtype,fill_value=0) )
            sparse.save_npz(savePath+'pos.{:03d}'.format(it+it0), sparse.COO.from_numpy(pos,idx_dtype=idx_dtype,fill_value=np.nan) )
        
        if saveFormat == 'npz':
            print("Numpy Compressed",end=end)
            np.savez_compressed(savePath+'charac_area_pos.{:03d}'.format(it+it0),charac=charac,area=area,pos=pos)
    
        print("Saved  ",end='')
        print(time.ctime()[-13:-5] ,str(round(time.time()-time0,1))+' s')
    

    


    
#%%
