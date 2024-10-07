#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:06:50 2024

@author: philippotn
"""

from numba import njit,prange
import numpy as np
floatype = np.float32
intype = np.int32

@njit(parallel=True)
def charac_opti_3D(objects,variables):
    # charac = (mean,std,min,max) of 3D objects for each levels, and each given 3D variables
    # area = area of objects per levels
    # pos = (x,y,std) mean position of clouds per levels, and its standard deviation 
    nz,nx,ny = objects.shape
    nobj = np.max(objects)+1
    charac = np.zeros((nobj,nz,len(variables),4),dtype=floatype)
    area = np.zeros((nobj,nz),dtype=intype)
    pos = np.zeros((nobj,nz,3),dtype=floatype)
    
    for h in prange(nz): # prange for parallelization of each levels
        for i in range(nx):
            for j in range(ny):
                obj = objects[h,i,j]
                area[obj,h] +=1
                
                if obj>0: # Compute mean position only for clouds (not for environment)
                    if area[obj,h]==1: 
                        pos[obj,h,0] = i
                        pos[obj,h,1] = j
                    else:
                        dx = i-pos[obj,h,0]
                        if nx-abs(dx)<abs(dx): # Periodic distance
                            dx = -np.sign(dx)* (nx-abs(dx))
                        dy = j-pos[obj,h,1]
                        if ny-abs(dy)<abs(dy):
                            dy = -np.sign(dy)* (ny-abs(dy))
                        pos[obj,h,0] = (pos[obj,h,0] + dx/area[obj,h])%nx # Periodic position thank to %
                        pos[obj,h,1] = (pos[obj,h,1] + dy/area[obj,h])%ny
                
                for v,variable in enumerate(variables): # Compute incrementaly (mean,std,min,max)
                    charac[obj,h,v,0] += variable[h,i,j]
                    charac[obj,h,v,1] += variable[h,i,j]**2
                    if charac[obj,h,v,2] > variable[h,i,j] or area[obj,h]==1:
                        charac[obj,h,v,2] = variable[h,i,j]
                    if charac[obj,h,v,3] < variable[h,i,j] or area[obj,h]==1:
                        charac[obj,h,v,3] = variable[h,i,j]
                        
    for h in prange(nz):
        for i in range(nx): # Loop to compute position variance from the previously computed mean position of clouds
            for j in range(ny):
                obj = objects[h,i,j]
                if obj>0:
                    dx = i-pos[obj,h,0]
                    if nx-abs(dx)<abs(dx):
                        dx = -np.sign(dx)* (nx-abs(dx))
                    dy = j-pos[obj,h,1]
                    if ny-abs(dy)<abs(dy):
                        dy = -np.sign(dy)* (ny-abs(dy))
                    pos[obj,h,2] += dx**2+dy**2
                        
    for h in prange(nz): 
        for obj in range(nobj):
            if area[obj,h]>0:
                pos[obj,h,2] = np.sqrt(pos[obj,h,2]/area[obj,h])
                for v in range(len(variables)):
                    charac[obj,h,v,0] /= area[obj,h] # Mean
                    charac[obj,h,v,1] = np.sqrt( charac[obj,h,v,1]/area[obj,h] - charac[obj,h,v,0]**2 ) # Standard deviation as sqrt(E[X^2] - E[X]^2)
            else:
                charac[obj,h,:,:] = np.nan # Outside of objects charac and pos are undefined 
                pos[obj,h,:] = np.nan
    return charac,area,pos

@njit(parallel=True)
def change_label(objects):
    nz,ny,nx = np.shape(objects)
    unique = np.unique(objects)
    old2new = np.zeros(unique[-1]+1,dtype=intype)
    for i,u in enumerate(unique):
        old2new[u] = i
        
    for j in prange(ny):
        for i in range(nx):
            for h in range(nz):
                obj = objects[h,i,j]
                if obj>0:
                    objects[h,i,j] = old2new[obj]