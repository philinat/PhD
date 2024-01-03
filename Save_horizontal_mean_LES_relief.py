#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:13:05 2023

@author: philippotn
"""

import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from numba import njit,prange
import os
import warnings
import sys
warnings.filterwarnings('ignore')
floatype = np.float32
intype = np.int32

# simu = 'AMOPL'
simu = 'A0W0V'
# simu = 'ATRI2'
# simu = 'ACON2'
# simu = 'AMO20'
# simu = 'CTRI2'
# simu = 'CFLAT'
# simu = 'CMO10'

orog = False

# userPath = '/cnrm/tropics/user/philippotn'
userPath = '/home/philippotn'
    
if userPath == '/home/philippotn':
    dataPath = userPath+'/Documents/SIMU_LES/'
    savePathVideo = userPath+'/Images/LES_'+simu+'/figures_maps/'
    sys.path.append(userPath+'/Documents/objects/src/')
    savePathObjects = userPath+'/Documents/objects/LES_'+simu+'/1M_thermals/'
else:
    dataPath = userPath+'/LES_'+simu+'/NO_SAVE/'
    savePathVideo = userPath+'/LES_'+simu+'/figures_maps/'
    sys.path.append(userPath+'/Code_LES/objects-master/src/')
    savePathObjects = userPath+'/LES_'+simu+'/characs_objects/1M_thermals/'

savePathMean = dataPath


if simu == 'AMOPL':
    seg = 'M200m'# ; lFiles = [dataPath + 'AMOPL.1.200m1.OUT.{:03d}.nc'.format(i) for i in range(1,241,1)] + [dataPath + 'AMOPL.1.200m2.OUT.{:03d}.nc'.format(i) for i in range(1,722,1)]
    lFiles = [dataPath + 'AMOPL.1.200m1.OUT.{:03d}.nc'.format(i) for i in [60,120,180,240]]+[dataPath + 'AMOPL.1.200m2.OUT.{:03d}.nc'.format(i) for i in [60,120,180,240,300,360,420,480,540,600,660,720]]
elif simu== 'A0W0V':
    seg = 'S200m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,962,60)]
elif simu== 'ATRI2':
    seg = 'S200m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,962,60)]
elif simu== 'ACON2':
    seg = 'S200m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,962,1)]
elif simu== 'AMO20':
    seg = 'M200m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in [1,61,121,181,241,301,361,421,481,541,601,661,699]]#range(1,961,1)]
elif simu== 'CTRI2':
    seg = 'M100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,872,1)]
elif simu== 'CFLAT':
    seg = 'S100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,872,1)]
elif simu== 'CMO10':
    seg = 'M100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(2,3,1)]

f0 = xr.open_dataset(lFiles[0],engine='h5netcdf')

nx1 = 1 ; nx2 = len(f0.ni)-1
ny1 = 1 ; ny2 = len(f0.nj)-1
nz1 = 1 ; nz2 = len(f0.level)

x = np.array(f0.ni)[nx1:nx2]
y = np.array(f0.nj)[ny1:ny2]
z = np.array(f0.level)[nz1:nz2]
z_ = np.array(f0.level_w)[nz1:nz2]
dz = z_[1:]-z_[:-1]
dz = np.append(dz,dz[-1])

Zm = np.array(f0.level)[nz1:]

dx = x[1]-x[0]
nt,nz,ny,nx = len(lFiles),len(z),len(y),len(x)

if orog:
    ZS = xr.open_dataset(dataPath+simu+'_init_'+seg+'_pgd.nc')['ZS'][ny1:ny2,nx1:nx2].data
    ZSmean = np.mean(ZS)
    MO = ZS>ZSmean # Mountain region (upper one)
    PL = np.logical_not(MO) # Plain region (lower one)
else:
    ZS = np.zeros((ny,nx))
    
Z = np.copy(z)

#%%
# import matplotlib.pyplot as plt
# import matplotlib
# hcm = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","saddlebrown","grey","w"])
# Hmax=2000
# extent = (0,nx*dx/1000,0,ny*dx/1000)
# fig, ax = plt.subplots(ncols=1,figsize=(12,10))
# im = ax.imshow(ZS, cmap=hcm,vmin=0,vmax=Hmax, origin='lower',extent=extent,interpolation='none')
# plt.colorbar(im,ax=ax,fraction=0.1, pad=0.01,orientation='vertical',extend=None)
# ax.contour(x/1000,y/1000,MO)
# ax.set_xlabel("x (km)")
# ax.set_ylabel("y")
#%%
from identification_methods import identify

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
        for i in prange(nx):
            for h in range(nz):
                obj = objects[h,i,j]
                if obj>0:
                    objects[h,i,j] = old2new[obj]
                    
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

@njit()
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

@njit()
def boundary_layer_height(z,ZS,rc,thv,rho,W,U,V):
    ny,nx = np.shape(rc[0])
    BLH = np.full((ny,nx),np.nan,dtype=floatype)
    THVBL = np.zeros((ny,nx),dtype=floatype) 
    WBL = np.zeros((ny,nx),dtype=floatype)
    UBLH = np.zeros((ny,nx),dtype=floatype)
    VBLH = np.zeros((ny,nx),dtype=floatype)
    cloud_base_mask = np.zeros((ny,nx),dtype=intype)
    ZTOP = z[-1]
    for j in range(ny):
        for i in range(nx):
            mean_thv = thv[0,j,i]
            mean_w = W[0,j,i]
            mass = z[0]*rho[0,j,i]
            for h in range(0,len(z)-1):
                if rc[h,j,i]>1e-6:
                    BLH[j,i] = ZS[j,i] + (ZTOP-ZS[j,i]) * z[h]/ZTOP
                    cloud_base_mask[j,i] = 1
                    break
                if thv[h,j,i] > mean_thv + 0.3:
                    BLH[j,i] = ZS[j,i] + (ZTOP-ZS[j,i]) * z[h]/ZTOP
                    break
                else:
                    layer_mass = (rho[h+1,j,i]+rho[h,j,i])/2 * (z[h+1]-z[h])
                    mean_thv = ( mean_thv*mass + (thv[h+1,j,i]+thv[h,j,i])/2 * layer_mass ) / ( mass+layer_mass)
                    mean_w = ( mean_w*mass + W[h+1,j,i] * layer_mass ) / ( mass+layer_mass)
                    mass += layer_mass
            THVBL[j,i] = mean_thv
            WBL[j,i] = mean_w
            UBLH[j,i] = U[h,j,i]
            VBLH[j,i] = V[h,j,i]
    return BLH,cloud_base_mask,THVBL,WBL,UBLH,VBLH

@njit()
def surface_anomaly(VAR,z,zmean,ZS):
    VARA = np.copy(VAR)
    ny,nx = np.shape(VARA)
    for j in range(ny):
        for i in range(nx):
            zVAR = ZS[j,i] + z[0]
            k = 0
            while zVAR>z[k+1]:
                k+=1
            dz = z[k+1]-z[k]
            VARA[j,i] -= ( zmean[k]*(z[k+1]-zVAR) + zmean[k+1]*(zVAR-z[k]) )/dz
    return VARA
@njit()
def horizontal_convergence(U,V):
    CONV = np.zeros_like(U)
    CONV[:,:-1,:] -= ( V[:,1:,:] - V[:,:-1,:] )/dx
    CONV[:,-1,:] -= ( V[:,0,:] - V[:,-1,:] )/dx
    CONV[:,:,:-1] -= ( U[:,:,1:] - U[:,:,:-1] )/dx
    CONV[:,:,-1] -= ( U[:,:,0] - U[:,:,-1] )/dx
    return CONV

X,Y = np.meshgrid(x,y)
ddot = 3*dx
xf = np.arange(x[0],x[-1]-ddot/2,ddot)
yf = np.arange(y[0],y[-1],ddot*np.sqrt(3)/2)
flat_X,flat_Y = np.meshgrid(xf,yf)
flat_X[::2,:] += ddot/2
flat_X = flat_X.ravel()
flat_Y = flat_Y.ravel()
@njit()
def get_flat_precip(x,y,flat_X,flat_Y,precip):
    n_flat = len(flat_X)
    flat_precip = np.zeros(n_flat)
    for i in range(n_flat):
        ix = np.argmin(np.abs(x-flat_X[i]))
        iy = np.argmin(np.abs(y-flat_Y[i]))
        flat_precip[i] = precip[iy,ix]
    return flat_precip

#%%
k = 2*np.pi*np.fft.fftfreq(nx,dx)
m = 2*np.pi*np.fft.fftfreq(ny,dx)
kk,mm = np.meshgrid(k,m)
k2 = kk**2+mm**2
del(kk,mm)
def wind_filter(U,V,lc):
    UV = U + 1j*V
    kc2 = (2*np.pi/lc)**2
    new_UV = np.fft.ifft2(np.exp(-k2/kc2/2)*np.fft.fft2(UV) )
    # new_UV = np.fft.ifft2((k2<kc2)*np.fft.fft2(UV) )
    return np.real(new_UV) , np.imag(new_UV)

# US = f0['UT'][0,nz1,ny1:ny2,nx1:nx2].data
# VS = f0['VT'][0,nz1,ny1:ny2,nx1:nx2].data

#%%
# def plot_US_VS(streamplot):
#     speed = np.sqrt(US**2+VS**2)
#     fig,axs = plt.subplots(ncols=3,constrained_layout=True)
#     wspace = 0.01
#     plt.subplots_adjust(left=wspace,right=1-wspace,bottom=0.1,top=0.95,wspace=wspace,hspace=wspace)
#     for i in range(3):
#         ax = axs[i]
#         ax.set_aspect("equal")
#         ax.set_xlim([x[0],x[-1]])
#         ax.set_ylim([y[0],y[-1]])
#         ax.set_xticks([])
#         ax.set_yticks([])
#         if streamplot:
#             ax.streamplot(X, Y, US, VS, density=3, color='k', linewidth=2*speed/10 , arrowsize = 0.5)
#             # ax.streamplot(X, Y, US, VS, density=3, color=speed,cmap='YlGn', linewidth=2*speed/10 , arrowsize = 0.5)
    
#     axs[0].pcolormesh(X,Y,US,vmin=-5,vmax=5,cmap='seismic')
#     axs[1].pcolormesh(X,Y,VS,vmin=-5,vmax=5,cmap='seismic')
#     axs[2].pcolormesh(X,Y,speed,vmin=0,vmax=5,cmap='rainbow')
    
#%%
# plot_US_VS(False)
# US,VS = wind_filter(US,VS,10000)
# plot_US_VS(True)

#%%
N=3
ft=20
fig,axs=plt.subplots(ncols=N,figsize=(16,9))#,constrained_layout=True)
wspace = 0.
plt.subplots_adjust(left=0.0+wspace,right=1-wspace,bottom=0.30, top=0.95,wspace=wspace,hspace=wspace)

fig.text(0.42,0.96,simu,fontsize=1.5*ft,ha='center')
text_time = fig.text(0.58,0.96,'{} h {:02d}'.format(int(f0.time.dt.hour),int(f0.time.dt.minute)),fontsize=1.5*ft,ha='center')

nstarts = 30
px,py = np.meshgrid(x[::nx//nstarts]/1000,y[::ny//nstarts]/1000)
start_points = np.transpose(np.array([px.flatten(),py.flatten()]))

def plot_Nmaps(ZS,U,V,THVAS,BLH,PRS,cloud_base_mask,WBL,UBLH,VBLH,name,legend=False):
    flat_precip = get_flat_precip(x,y,flat_X,flat_Y,PRS)
    select_precip = flat_precip > 10**-4 # 1 mm of rain in 1 min
    
    
    
    for i in range(N):
        ax = axs[i]
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim([x[0]/1000,x[-1]/1000])
        ax.set_ylim([y[0]/1000,y[-1]/1000])
        # ax.set_xticks([])
        if i>=0: ax.set_yticks([])
        plt.xticks(fontsize=ft*2/3)
        # ax.set_xlabel('km',fontsize=ft,rotation=0)
        if orog:
            CS = ax.contour(x/1000,y/1000,ZS, levels=[np.percentile(ZS,5),np.mean(ZS),np.percentile(ZS,95)],linewidths=1.,alpha=0.5,colors=['k'] , zorder=2)
            contour_fmt='%d'
            plt.clabel(CS, CS.levels, inline=True, inline_spacing=0, fmt=contour_fmt, fontsize=ft/3)
        
        ax.scatter(flat_X[select_precip]/1000, flat_Y[select_precip]/1000, s= flat_precip[select_precip]*2e2, c='black',zorder=100)
        
    cb_length = 0.9 /N
    cb_width = 0.03
    cb_posy = 0.20
    shrink = 1.
    
    ax = axs[0]
    T_cf = ax.pcolormesh(x/1000,y/1000,THVAS,vmin=-4, vmax=4,cmap='RdBu_r',shading='gouraud')
    if legend:
        cax = fig.add_axes([1/N/2-cb_length/2, cb_posy, cb_length, cb_width])
        T_cb = plt.colorbar(T_cf,cax=cax,shrink=shrink,orientation = "horizontal",extend='both',format='%.1f')
        # T_cb.set_ticks([-5,-2,-1,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1.,2.,5.])
        T_cb.ax.tick_params(labelsize=ft*2/3)
        T_cb.ax.set_title("$\\theta _v ^{\prime}$ and wind at 5m AGL",fontsize=ft)
        T_cb.set_label(' K',fontsize=ft,rotation=0)
    
    # ax.contour(x/1000,y/1000,cloud_base_mask, levels=[0.5],linewidths=2,alpha=1,colors=['grey'])#,colors=[CTcm( (z[cloud_top[obj]]/1000 -ct_lev[0])/ct_lev[-1] )])
    
    US,VS = wind_filter(U[0],V[0],1e4)
    ax.streamplot(x/1000,y/1000, US, VS, density=3, color='k', linewidth=2*np.sqrt(US**2+VS**2)/10 , arrowsize = 0.5,start_points=start_points)
    

    ax = axs[1]
    BLH_cf = ax.pcolormesh(x/1000,y/1000,BLH,vmin=0.,vmax=4000.,cmap='rainbow',shading='gouraud')
    if legend:
        cax = fig.add_axes([3/N/2-cb_length/2, cb_posy, cb_length, cb_width])
        BLH_cb = plt.colorbar(BLH_cf,cax=cax,shrink=shrink,orientation = "horizontal",extend='max',format='%d')
        BLH_cb.set_ticks([0,1000,2000,3000,4000])
        BLH_cb.ax.tick_params(labelsize=ft*2/3)
        BLH_cb.ax.set_title("Top of the mixed layer",fontsize=ft)
        BLH_cb.set_label(' m',fontsize=ft,rotation=0)
    
    ax.contour(x/1000,y/1000,cloud_base_mask, levels=[0.5],linewidths=1,alpha=1,colors=['k'])#,colors=[CTcm( (z[cloud_top[obj]]/1000 -ct_lev[0])/ct_lev[-1] )])
    US,VS = wind_filter(UBLH,VBLH,1e4)
    ax.streamplot(x/1000,y/1000, US, VS, density=3, color='k', linewidth=2*np.sqrt(US**2+VS**2)/10 , arrowsize = 0.5,start_points=start_points)
    
    
    ax = axs[2]
    W_cf = ax.pcolormesh(x/1000,y/1000,WBL,vmin=-5, vmax=5,cmap='seismic',shading='gouraud')
    if legend:
        cax = fig.add_axes([5/N/2-cb_length/2, cb_posy, cb_length, cb_width])
        W_cb = plt.colorbar(W_cf,cax=cax,shrink=shrink,orientation = "horizontal",extend='both',format='%.1f')
        # W_cb.set_ticks([-5,-2,-1,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1.,2.,5.])
        W_cb.ax.tick_params(labelsize=ft*2/3)
        W_cb.ax.set_title("Mean W in the mixed layer",fontsize=ft)
        W_cb.set_label(' m/s',fontsize=ft,rotation=0)
    
    if legend:
        cax=fig.add_axes([2/N-0.1,0.08, 0.2, 0.03])
        cax.set_xlim([0.5,4.5])
        cax.set_ylim([0.,2.])
        cax.set_yticks([])
        cax.set_xticks([1,2,3,4])
        cax.set_xticklabels(['0.1','1','10','100'],size=ft*2/3)
        # cax.xticks.label.set_fontsize(2/3*ft) 
        cax.scatter([1,2,3,4],[1,1,1,1],s=0.5*np.array([0.1,1,10,100]), c='black')
        cax.set_xlabel('mm / min',fontsize=ft)
        cax.set_title('Rain at the surface',fontsize=ft)
        
        cax=fig.add_axes([1/N-0.1,0.08, 0.2, 0.03])
        cax.set_xlim([0.5,4.5])
        cax.set_ylim([0.,2.])
        cax.set_yticks([])
        cax.set_xticks([1,2,3,4])
        cax.set_xticklabels(['1','5','10','20',],size=ft*2/3)
        US,VS = np.array([[1,5,10,20],[1,5,10,20],[1,5,10,20]]) , np.zeros((3,4))
        cax.streamplot(np.arange(1,5),np.arange(3),US,VS, density=3, color='k', linewidth=2*np.sqrt(US**2+VS**2)/10 , arrowsize = 0.5,start_points=np.array([[2.5,1.]]) )
        cax.set_xlabel('m/s',fontsize=ft)
        cax.set_title('Wind speed',fontsize=ft)
        
    if not(legend):
        text_time.set_text('{} h {:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute)))

    if not os.path.exists(savePathVideo+'maps_video/'): os.makedirs(savePathVideo+'maps_video/') ; print('Directory created !')
    
    plt.savefig(savePathVideo+'maps_video/maps_video_{:02d}h{:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute))+simu+'.png')
    
    print(' ---- Saved {:02d}h{:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute)))
    
#%%
nvar = 9
nflux = 4
# timeH = np.zeros(nt)
data_mean = np.full((nvar,nt,nz),np.nan,dtype=floatype)
precip_mean = np.zeros(nt,dtype=floatype)
if orog:
    data_flux_MO = np.full((nflux,nt,nz),np.nan,dtype=floatype)
    data_mean_MO = np.full((nvar,nt,nz),np.nan,dtype=floatype)
    data_mean_PL = np.full((nvar,nt,nz),np.nan,dtype=floatype)
    precip_mean_MO = np.zeros(nt,dtype=floatype)
    precip_mean_PL = np.zeros(nt,dtype=floatype)

times = f0.time.data
it0 = 1
img = []
for it,fname in enumerate(lFiles):
    time0 = time.time()
    f = xr.open_dataset(fname,engine='h5netcdf')
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
    RT = RV+RC+RP
    P = npf('PABST')
    W = npf('WT')
    U = npf('UT')
    V = npf('VT')
    TR = npf('SVT001')
    PRS = np.array(f.ACPRRSTEP[0,ny1:ny2,nx1:nx2],dtype=floatype)
    
    print(' Read :',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
    # T = TH * (P/100000.)**(2/7)
    THV = TH * (1.+ 1.61*RV)/(1.+ RT )
    Rd = 287.04     # J/kg/K 
    RHO = P/Rd/ (THV * (P/100000.)**(2/7))
    CONV = horizontal_convergence(U,V)
    BLH,cloud_base_mask,THVBL,WBL,UBLH,VBLH = boundary_layer_height(z,ZS,RC,THV,RHO,W,U,V)
    
    print(' Compute variables:',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
    listVarNames = ['SVT001','WT']#[]#
    def thermalMask(dictVars):
        TR = dictVars['SVT001'][0,nz1:nz2,ny1:ny2,nx1:nx2]
        W = dictVars['WT'][0,nz1:nz2,ny1:ny2,nx1:nx2]
        TR_std = np.std(TR,axis=(1,2))
        TR_mean = np.mean(TR,axis=(1,2))
        TR_stdmin = 0.05/(z+dz/2) * np.cumsum(dz*TR_std)
        TR_threshold = TR_mean+np.maximum(TR_std,TR_stdmin)
        
        mask = np.zeros((1,nz+1,ny+2,nx+2),dtype='bool')
        mask[0,nz1:nz2,ny1:ny2,nx1:nx2] = np.logical_and( TR>TR_threshold.reshape(nz,1,1) , W>0 )
        mask[0,:,0,:] = mask[0,:,-2,:]
        mask[0,:,-1,:] = mask[0,:,1,:]
        mask[0,:,:,0] = mask[0,:,:,-2]
        mask[0,:,:,-1] = mask[0,:,:,1]
        return mask
    
    objects,_ = identify(fname, listVarNames, thermalMask, name="tracer_updraft",delete=10,write=False,rename=False)
    print(' Identify:',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
    objects = np.array(objects[0,nz1:nz2,ny1:ny2,nx1:nx2],dtype=np.int32)
    change_label(objects)
    # variables = np.array([U,V,W,CONV,THV,P,TR,RC,RP,RT,W*RHO*RT],dtype=np.float32 ) 
    variables = [U,V,W,CONV,THV,P,TR,RC,RP,RT,W*RHO*RT]
    charac,area,pos = charac_opti_3D(objects, variables )
    print(' Charac:',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
    if not os.path.exists(savePathObjects): os.makedirs(savePathObjects) ; print('savePath directory created !')
    np.savez_compressed(savePathObjects+'objects_charac_area_pos.{:03d}'.format(it+it0),objects=objects,charac=charac,area=area,pos=pos)
    print(' Save charac:',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
    if orog:
        TH_ = interp_on_altitude_levels( TH,Z,Zm,ZS)
        RV_ = interp_on_altitude_levels(RV,Z,Zm,ZS)
        THV_ = interp_on_altitude_levels( THV,Z,Zm,ZS)
        RC_ = interp_on_altitude_levels(RC,Z,Zm,ZS)
        P_ = interp_on_altitude_levels( P,Z,Zm,ZS)
        TR_ = interp_on_altitude_levels(TR,Z,Zm,ZS)
        W_ = interp_on_altitude_levels(W,Z,Zm,ZS)
        U_ = interp_on_altitude_levels(U,Z,Zm,ZS)
        V_ = interp_on_altitude_levels(V,Z,Zm,ZS)
    else:
        TH_=TH ; RV_=RV ; THV_=THV ; RC_=RC ; P_=P ; TR_=TR ; W_=W ; U_=U ; V_=V
    print('Interpolations on altitude levels :',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
    precip_mean[it] = np.mean(PRS)
    data_mean[0,it] = np.nanmean(TH_,axis=(1,2)) # K
    data_mean[1,it] = np.nanmean(RV_,axis=(1,2)) # g/kg
    data_mean[2,it] = np.nanmean(THV_,axis=(1,2)) # K
    data_mean[3,it] = np.nanmean( RC_>1e-6 ,axis=(1,2)) # cloud fraction
    data_mean[4,it] = np.nanmean(P_,axis=(1,2)) # Pa
    data_mean[5,it] = np.nanmean(TR_,axis=(1,2))
    data_mean[6,it] = np.nanmean(W_,axis=(1,2)) # m/s
    data_mean[7,it] = np.nanmean(U_,axis=(1,2)) # m/s
    data_mean[8,it] = np.nanmean(V_,axis=(1,2)) # m/s
    
    if orog:
        data_flux_MO[:,it,:] = MO_flux(MO,U,V,RHO,RV,RT,TH)
        
        precip_mean_MO[it] = np.mean(PRS[MO])
        data_mean_MO[0,it] = np.nanmean(TH_[:,MO],axis=1) # K
        data_mean_MO[1,it] = np.nanmean(RV_[:,MO],axis=1) # g/kg
        data_mean_MO[2,it] = np.nanmean(THV_[:,MO],axis=1) # K
        data_mean_MO[3,it] = np.nanmean( RC_[:,MO]>1e-6,axis=1) # cloud fraction
        data_mean_MO[4,it] = np.nanmean(P_[:,MO],axis=1) # Pa
        data_mean_MO[5,it] = np.nanmean(TR_[:,MO],axis=1)
        data_mean_MO[6,it] = np.nanmean(W[:,MO],axis=1) # m/s
        data_mean_MO[7,it] = np.nanmean(U_[:,MO],axis=1) # m/s
        data_mean_MO[8,it] = np.nanmean(V_[:,MO],axis=1) # m/s
        
        precip_mean_PL[it] = np.mean(PRS[PL])
        data_mean_PL[0,it] = np.nanmean(TH_[:,PL],axis=1) # K
        data_mean_PL[1,it] = np.nanmean(RV_[:,PL],axis=1) # g/kg
        data_mean_PL[2,it] = np.nanmean(THV_[:,PL],axis=1) # K
        data_mean_PL[3,it] = np.nanmean( RC_[:,PL]>1e-6,axis=1) # cloud fraction
        data_mean_PL[4,it] = np.nanmean(P_[:,PL],axis=1) # Pa
        data_mean_PL[5,it] = np.nanmean(TR_[:,PL],axis=1)
        data_mean_PL[6,it] = np.nanmean(W[:,PL],axis=1) # m/s
        data_mean_PL[7,it] = np.nanmean(U_[:,PL],axis=1) # m/s
        data_mean_PL[8,it] = np.nanmean(V_[:,PL],axis=1) # m/s
    
    print('Nanmeans :',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
    THVAS = surface_anomaly(THV[0],z,data_mean[2,it],ZS)
    
    name = savePathVideo+'maps_video/maps_video_{:02d}h{:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute))+simu+'.png'
    if it == 0:
        plot_Nmaps(ZS,U,V,THVAS,BLH,PRS,cloud_base_mask,WBL,UBLH,VBLH,name,legend=True)
    else:
        plot_Nmaps(ZS,U,V,THVAS,BLH,PRS,cloud_base_mask,WBL,UBLH,VBLH,name,legend=False)
    img.append(name)
    
    print('Make and save maps :' ,str(round(time.time()-time0,1))+' s')

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
        "U": (("time", "z"), data_mean[7]),
        "V": (("time", "z"), data_mean[8]),
        "PR": (("time",),precip_mean),
        
        "MO": (("y", "x"), MO),
        "TH_MO": (("time", "z"), data_mean_MO[0]),
        "RV_MO": (("time", "z"), data_mean_MO[1]),
        "THV_MO": (("time", "z"), data_mean_MO[2]),
        "CF_MO": (("time", "z"), data_mean_MO[3]),
        "P_MO": (("time", "z"), data_mean_MO[4]),
        "TR_MO": (("time", "z"), data_mean_MO[5]),
        "W_MO": (("time", "z"), data_mean_MO[6]),
        "U_MO": (("time", "z"), data_mean_MO[7]),
        "V_MO": (("time", "z"), data_mean_MO[8]),
        "PR_MO": (("time",),precip_mean_MO),
        
        "PL": (("y", "x"), PL),
        "TH_PL": (("time", "z"), data_mean_PL[0]),
        "RV_PL": (("time", "z"), data_mean_PL[1]),
        "THV_PL": (("time", "z"), data_mean_PL[2]),
        "CF_PL": (("time", "z"), data_mean_PL[3]),
        "P_PL": (("time", "z"), data_mean_PL[4]),
        "TR_PL": (("time", "z"), data_mean_PL[5]),
        "W_PL": (("time", "z"), data_mean_PL[6]),
        "U_PL": (("time", "z"), data_mean_PL[7]),
        "V_PL": (("time", "z"), data_mean_PL[8]),
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
        "U": (("time", "z"), data_mean[7]),
        "V": (("time", "z"), data_mean[8]),
        "PR": (("time",),precip_mean)
        },
        coords={"time": times,
            "z": Z,
            "y": y,
            "x": x},
)

if not os.path.exists(savePathMean): os.makedirs(savePathMean) ; print('savePath directory created !')
ds.to_netcdf(savePathMean+simu+'_'+seg+'_mean.nc', encoding={"time": {'units': 'days since 1900-01-01'}})

#%%
from moviepy.editor import ImageSequenceClip

# duration = 2
# m0 = 60*12 + 0
# m0 = 60*9 + 30
# img = [savePathVideo+'maps_video/maps_video_{:02d}h{:02d}'.format((m0+m)//60,(m0+m)%60)+simu+'.png' for m in range(len(lFiles))]
# img = [savePath+'maps_video/maps_video_{:03d}h'.format(h)+simu+'.png' for h in range(1,121)]

# clips = [ImageClip(m).set_duration(2) for m in img]
concat_clip = ImageSequenceClip(img,  fps=10)
concat_clip.write_videofile(savePathVideo+'video_3maps_'+simu+'.mp4')

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

#%%
# def plot_profiles(z,data_profile):
#     import matplotlib.pyplot as plt
#     fig,axs = plt.subplots(ncols=len(data_profile))
#     for it,tt in enumerate(times):
#         for i in range(len(data_profile)):
#             axs[i].plot(data_profile[i,it],z,label=tt)
#     axs[0].legend()
    
# plot_profiles(z,data_mean_MO)
# plot_profiles(z,data_mean_PL)
# plot_profiles(z,data_flux_MO)

#%%
# for it,tt in enumerate(times):
#     for i in range(4):
#         print( np.sum(dz*data_flux_MO[i,it]) / np.sum(dz*np.abs(data_flux_MO[i,it])) )
        
#%%
# fig,ax = plt.subplots()
# ax2 = ax.twiny()
# for it,tt in enumerate(times):
#     ax.plot(data_mean_MO[2,it]-data_mean_PL[2,it] , z)
#     ax2.plot(data_flux_MO[0,it] , ZSmean + (1-ZSmean/z[-1]) * z)
    
#%%
# TR_max = np.max(TR,axis=(1,2))
# TR_std = np.std(TR,axis=(1,2))
# TR_mean = np.mean(TR,axis=(1,2))
# TR_stdmin = 0.05/(z+dz/2) * np.cumsum(dz*TR_std)
# TR_threshold = TR_mean+np.maximum(TR_std,TR_stdmin)

# plt.plot(TR_mean,z,label='mean')
# plt.plot(TR_max,z,label='max')
# plt.plot(TR_std,z,label='std')
# plt.plot(TR_stdmin,z,label='stdmin')
# plt.plot(TR_threshold,z,label='threshold')

# plt.plot(np.max(W*TR,axis=(1,2)),z,label='Wmax')
# thermals = np.logical_and( TR>TR_threshold.reshape(nz,1,1) , W>0 )
# plt.plot(np.mean(thermals,axis=(1,2)),z,label='thermal fraction')

# plt.legend()

#%%
# list_k = [0,10,20,30]
# fig,ax = plt.subplots(ncols=len(list_k))
# for i,k in enumerate(list_k):
#     ax[i].pcolormesh(x,y,TR[k],cmap='Greens')
#     ax[i].contour(x,y,np.logical_and(TR[k]>TR_threshold[k],W[k]>0.))
    
#%%

# listVarNames = []#['SVT001','WT']
# def thermalMask(dictVars):
#     # TR = dictVars['SVT001'][0,nz1:nz2,ny1:ny2,nx1:nx2]
#     # W = dictVars['WT'][0,nz1:nz2,ny1:ny2,nx1:nx2]
#     TR_std = np.std(TR,axis=(1,2))
#     TR_mean = np.mean(TR,axis=(1,2))
#     TR_stdmin = 0.05/(z+dz/2) * np.cumsum(dz*TR_std)
#     TR_threshold = TR_mean+np.maximum(TR_std,TR_stdmin)
    
#     mask = np.zeros((1,nz+1,ny+2,nx+2),dtype='bool')
#     mask[0,nz1:nz2,ny1:ny2,nx1:nx2] = np.logical_and( TR>TR_threshold.reshape(nz,1,1) , W>0 )
#     mask[0,:,0,:] = mask[0,:,-2,:]
#     mask[0,:,-1,:] = mask[0,:,1,:]
#     mask[0,:,:,0] = mask[0,:,:,-2]
#     mask[0,:,:,-1] = mask[0,:,:,1]
#     return mask
    
# for fname in lFiles[:]:
#     print("\n "+fname)
#     objects,_ = identify(fname, listVarNames, thermalMask, name="tracer_updraft",delete=20,write=False,rename=True)
#     #%%
#     f = xr.open_dataset(fname,engine='h5netcdf')
#     f["tracer_updraft"] = (("time", "level", "nj", "ni"),objects)
#     f["tracer_updraft"].attrs['units'] = 'label'
#     f["tracer_updraft"].attrs['comment'] = 'Objects defined from tracer anomaly and W>0 like in Weinkaemmerer2023'
    
#     f.to_netcdf(fname,mode='a',engine='h5netcdf',encoding={"time": {'units': 'days since 1900-01-01'}})