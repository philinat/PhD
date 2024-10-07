#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:12:15 2022

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

simu = 'AMMA'
# simu = 'AMA0W'
# simu = 'LBA'

# you need a tFile to get 
if simu == 'AMMA':
    path = '/cnrm/tropics/user/couvreux/POUR_NATHAN/AMMA/SIMU_LES/'
    lFiles = [path + 'AMMH3.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(1,361)]
    
elif simu == 'AMA0W':
    path = '/cnrm/tropics/user/philippotn/LES_AMA0W/SIMU_LES/'
    lFiles = [path + 'AMA0W.1.R6H1M.OUT.{:03d}.nc'.format(i) for i in range(1,361)]
    
elif simu == 'LBA':
    path = '/cnrm/tropics/user/philippotn/LES_LBA/SIMU_LES/'
    lFiles = [path + 'LBA__.1.10H1M.OUT.{:03d}.nc'.format(i) for i in range(1,361)]
    # lFiles = [path + 'LBA_2.1.10H1M.OUT.{:03d}.nc'.format(i) for i in range(1,241)]
    
it0 = 1 # Time index of the first file  
savePath = '/cnrm/tropics/user/philippotn/LES_'+simu+'/characs_objects/1M_clouds/'

if not os.path.exists(savePath):
    os.makedirs(savePath)
    print('Directory created !')
    
#%% Physic functions for Meso-NH data


g = 9.81 # m/s**2 
R = 287.06 # J/kg/K
Cp = 7/2*R # J/kg/K #1004.71
P0 = 100000 # Pa
T0 = 273.16 # K
Rv = 461.525 # J/kg/K
rho0 = P0/R/T0 # kg/m^3
rho_lw = 1000 # kg/m^3
rho_i = 900 # kg/m^3

f = xr.open_dataset(lFiles[0])

x = np.array(f.ni,dtype=floatype)
dx = x[1]-x[0]

def npf(v,dtype=floatype, order='C'):
    return np.array(f[v][0,1:-1,1:-1,1:-1],dtype=dtype, order=order)
P = npf('PABST')
theta = npf('THT')
rv = npf('RVT')
T = theta*(P/100000.)**(R/Cp)
rho_d = np.mean(P/R/T/(1+rv),axis=(1,2),keepdims=True)
del(P,theta,rv,T)

def Lv(T):
    return 2.5008e6 + (4*Rv - 4.218e3)*(T-T0)
def Ls(T):
    return 2.8345e6 + (4*Rv - 2.106e3)*(T-T0)
def temperature(theta,P):
    return theta*(P/100000.)**(R/Cp)
def RainFallSpeed(rr,rho_d):
    N0 = 8e6 ; b=0.8 ; a=842. #; rholw = 1000
    gamma = math.gamma(b+4)#17.8378619818136 # 
    return np.minimum( 7., a/6*gamma*(rho0/rho_d)**0.4 * (rho_d*rr/np.pi/rho_lw/N0)**(b/4) )
def G1(p):
    return math.gamma(1.+p)
def SnowFallSpeed(rs,rho_d):
    a=0.02 ; b=1.9 ; c=5.1 ; d=0.27 ; x=1. ; C=5.
    return C*a*c*G1(b+d) *(rho0/rho_d)**0.4 / (a*C*G1(b))**((b+d-x)/(b-x)) * (rho_d*rs)**(d/(b-x))
def GraupelFallSpeed(rg,rho_d):
    a=19.6 ; b=2.8 ; c=124. ; d=0.66 ; x=-0.5 ; C=5e5
    return C*a*c*G1(b+d) *(rho0/rho_d)**0.4 / (a*C*G1(b))**((b+d-x)/(b-x)) * (rho_d*rg)**(d/(b-x))
def RainFallSpeed_test(rr,rho_d):
    a=524 ; b=3. ; c=842 ; d=0.8 ; x=-1. ; C=8e6
    return np.minimum( 7., C*a*c*G1(b+d) *(rho0/rho_d)**0.4 / (a*C*G1(b))**((b+d-x)/(b-x)) * (rho_d*rr)**(d/(b-x)) )
def G3(p):
    return math.gamma(3.+p/3.)/2.
def IceFallSpeed(ri,rho_d):
    a=0.82 ; b=2.5 ; c=800. ; d=1.0
    Nci_rho_r = 0.25/np.pi/rho_i * np.maximum( 5e4 , - 0.1532e6 - 0.021454e6*np.log(rho_d*ri+1e-15))**3
    return Nci_rho_r*a*c * G3(b+d) * (1/a/Nci_rho_r/G3(b))**((b+d)/b) * (rho0/rho_d)**0.4
def horizontal_convergence(u,v):
    conv = np.zeros_like(u)
    conv[:,1:,1:] = ( u[:,1:,1:]+u[:,1:,:-1] - u[:,:-1,1:]-u[:,:-1,:-1] + v[:,1:,1:]+v[:,:-1,1:] - v[:,1:,:-1]-v[:,:-1,:-1] )/2/dx
    conv[:,1:-1,1:-1] = (conv[:,1:-1,1:-1]+conv[:,1:-1,2:]+conv[:,2:,1:-1]+conv[:,2:,2:])/4
    conv[:,0,1:-1] = conv[:,-2,1:-1]
    conv[:,-1,1:-1] = conv[:,1,1:-1]
    conv[:,:,0] = conv[:,:,-2]
    conv[:,:,-1] = conv[:,:,1]
    return -conv

#%% Main functions
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
def extend_to_surface(objects): # Extend verticaly clouds objects from cloud base to the surface
    nz,ny,nx = np.shape(objects)
    z_count = np.zeros(nz)
    for j in prange(ny): # Loop to count the number of cloud having the same base altitude ( == histogram of cloud base )
        for i in prange(nx):
            for h in range(nz):
                if objects[h,j,i]>0:
                    z_count[h] += 1
                    break
                
    for h in range(1,nz-4): # The mean cloud base level is selected as the first spike of the cloud base histogram ( we verify that there is no bigger spike in the next 3 levels to avoid errors )
        if z_count[h]>100 and z_count[h+1] < z_count[h] and z_count[h+2] < z_count[h] and z_count[h+3] < z_count[h]:
            CL = h # CL = condensation level
            break
        
    for j in prange(ny): # Loop to extend the cloud objects to the surface ( if the cloud base is close enough to the mean condensation level )
        for i in prange(nx):
            h=0
            while h<CL+3:
                h+=1
                if objects[h,j,i]>0 :
                    obj = objects[h,j,i]
                    for iz in range(h):
                        objects[iz,j,i] = obj
                    break
                
@njit(parallel=True)
def change_label(objects):
    nz,ny,nx = np.shape(objects)
    unique = np.unique(objects)
    old2new = np.zeros(unique[-1]+1,dtype=intype)
    for i,u in enumerate(unique):
        old2new[u] = i
        
    for j in prange(ny):
        for i in prange(nx):
            for h in range(nz):
                obj = objects[h,i,j]
                if obj>0:
                    objects[h,i,j] = old2new[obj]

#%% Main loop : prepare your 3D variables and characterize your objects

if minimal_setting_for_time_tracking:
    for it,file in enumerate(lFiles[:]):
        time0 = time.time()
        # print(file)
        print('{:03d}'.format(it+1),end=' ')
        print("Reading file",end=end)
        f = xr.open_dataset(file)
        def npf(v,dtype=floatype, order='C'):
            return np.array(f[v][0,1:-1,1:-1,1:-1],dtype=dtype, order=order)
        
        print("Objects",end=end)
        objects = npf('clouds',dtype=intype)
        change_label(objects)
        
        print("Variables",end=end)
        variables = np.array([npf('UT'),npf('VT')],dtype=floatype)
        
        print("Charac",end=end)
        charac,area,pos = charac_opti_3D(objects,variables)
        
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
    
else:
    for it,file in enumerate(lFiles[:]):
        time0 = time.time()
        # print(file)
        print('{:03d}'.format(it+1),end=' ')
        print("Reading file",end=end)
        f = xr.open_dataset(file)
        def npf(v,dtype=floatype):
            return np.array(f[v][0,1:-1,1:-1,1:-1],dtype=dtype, order='C')
        
        w = npf('WT')
        
        print("Temperature",end=end)
        theta = npf('THT')
        P = npf('PABST')
        anom_P = P - np.mean(P,axis=(1,2),keepdims=True)
        T = theta*(P/100000.)**(R/Cp)
        
        print("Water",end=end)
        rv , rc , ri , rr , rs , rg = npf('RVT') , npf('RCT') , npf('RIT') , npf('RRT') , npf('RST') , npf('RGT')
        rt = rv+rc+ri+rr+rs+rg
        precip = rr+rs+rg
        
        print("Water flux",end=end)
        rho_d = np.mean(P/R/T/(1+rv),axis=(1,2),keepdims=True)
        flux_water0 = w*rho_d*rt
        flux_water = np.copy(flux_water0)
        flux_water -= RainFallSpeed(rr,rho_d) *rho_d*rr
        flux_water -= IceFallSpeed(ri,rho_d) *rho_d*ri
        flux_water -= SnowFallSpeed(rs,rho_d) *rho_d*rs
        flux_water -= GraupelFallSpeed(rg,rho_d) *rho_d*rg
 
        print("Buoyancy",end=end)
        thetav = theta * (1+ 1.61*rv)/(1+rt)
        anom_thetav = thetav - np.mean(thetav,axis=(1,2),keepdims=True)
        
        print("Theta li flux",end=end)
    #    thetali = theta - theta/T /Cp * ( Lv(T)*(rc+rr) + Ls(T)*(ri+rs+rg) )
        thetali = theta - theta/T /Cp * ( Lv(T)*(rc) + Ls(T)*(ri) )
        flux_thetali = w*(thetali - np.mean(thetali,axis=(1,2),keepdims=True))
        
        print("Wind convergence",end=end)
        U = np.array(f['UT'][0,1:-1,:,:],dtype=floatype, order='C')
        V = np.array(f['VT'][0,1:-1,:,:],dtype=floatype, order='C')
        conv = horizontal_convergence(V,U)
        U = np.array(U[:,1:-1,1:-1], order='C')
        V = np.array(V[:,1:-1,1:-1], order='C')
        conv = np.array(conv[:,1:-1,1:-1], order='C')
        
        print("Objects",end=end)
        objects = npf('clouds',dtype=intype)
        change_label(objects)
        extend_to_surface(objects)
        
        print("Variables",end=end)
        
        variables = [U,V,conv,w,anom_thetav,anom_P,rt,rv,precip,flux_water0,flux_water,flux_thetali]
        # If it doesn't work with the reflected list above, use the line below : ( almost double the need for RAM !! )
        # variables = np.array([U,V,conv,w,anom_thetav,anom_P,rt,rv,precip,flux_water0,flux_water,flux_thetali],dtype=floatype)
        
        print("Charac",end=end)
        charac,area,pos = charac_opti_3D(objects,variables)
        
        if saveFormat == 'npz':
            print("Numpy Compressed",end=end)
            np.savez_compressed(savePath+'charac_area_pos.{:03d}'.format(it+it0),charac=charac,area=area,pos=pos)
        
        
        if saveFormat == 'sparse':
            print("Sparse",end=end)
            sparse.save_npz(savePath+'charac.{:03d}'.format(it+it0), sparse.COO.from_numpy(charac,idx_dtype=idx_dtype,fill_value=np.nan) )
            sparse.save_npz(savePath+'area.{:03d}'.format(it+it0), sparse.COO.from_numpy(area,idx_dtype=idx_dtype,fill_value=0) )
            sparse.save_npz(savePath+'pos.{:03d}'.format(it+it0), sparse.COO.from_numpy(pos,idx_dtype=idx_dtype,fill_value=np.nan) )
        
        
        
        print("Saved  ",end='')
        print(time.ctime()[-13:-5] ,str(round(time.time()-time0,1))+' s')
    


    
#%%
