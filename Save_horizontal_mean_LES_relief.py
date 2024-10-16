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
import matplotlib
# import matplotlib.colors as mcolors
ZScm = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","saddlebrown","grey","w"])
Wcm = matplotlib.colors.LinearSegmentedColormap.from_list('seismic2',[(0,(0.2,0.,0.3)),(0.25,(0,0,1)),(0.28,(0,0.2,1)),(0.45,(0,0.9,1)),(0.5,(1,1,1)),(0.55,(1,0.9,0)),(0.73,(1,0.2,0)),(0.75,(1,0,0)),(1,(0.3,0.,0.2))])
    
from numba import njit,prange
import os
import warnings
import sys
import sparse


from Module_LES_objects import change_label, charac_opti_3D
from Module_LES import Md,Mv,Rd,Cp,Lv,g,p0
from Module_LES import RH_water_ice, dewpoint
from Module_LES import AL_lin_interp, AL_objects, AGL_lin_interp, AGL_lin_anomaly, boundary_layer_diags, surface_layer_diags, horizontal_convergence , MO_flux, AL_SL

warnings.filterwarnings('ignore')
floatype = np.float64
intype = np.int32

make_charac = False
make_maps = False ; img_maps_names = ['gif_BLH','gif_WBL','gif_THVA2m'] # img_maps_names = ['video_3maps_BL','video_3maps_SL']
make_hist2D = False
make_sparses = False
make_hori_mean = True
# use_thermals = True

orog = True
dates = False

# simu = 'AMOPL'
# simu = 'A0W0V'
# simu = 'ATRI2'
# simu = 'ACON2'
# simu = 'AMO20'
# simu = 'CTRI2'
# simu = 'CFLAT'
# simu = 'CMO10'

# simu = 'E600M'
# simu = 'E620M'

# simu = 'DFLAT'
# simu = 'DF500'
# simu = 'DMO10'
# simu = 'DTRI2'
simu = 'DMOPL'





# userPath = '/cnrm/tropics/user/philippotn'
userPath = '/home/philippotn'
    
if userPath == '/home/philippotn':
    if simu[0] in ['A','C']: dataPath = userPath+'/Documents/SIMU_LES/'
    elif simu[0] in ['D']: dataPath = userPath+'/Documents/NO_SAVE/'
    elif simu[0] in ['E']: dataPath = userPath+'/Documents/Canicule_Août_2023/'+simu+'/'
    # dataPath = userPath+'/Documents/SIMU_LES/'
    savePathFigures = userPath+'/Images/LES_'+simu+'/'
    sys.path.append(userPath+'/Documents/objects/src/')
    savePathObjects = userPath+'/Documents/objects/LES_'+simu+'/1M_thermals/'
    savePathSkyView = userPath+'/Documents/SkyView/data_sparse/'
    
else:
    dataPath = userPath+'/LES_'+simu+'/NO_SAVE/'
    savePathFigures = userPath+'/LES_'+simu+'/figures/'
    sys.path.append(userPath+'/Code_LES/objects-master/src/')
    savePathObjects = userPath+'/LES_'+simu+'/characs_objects/1M_thermals/'
    savePathSkyView = userPath+'/SkyView/data_sparse/'
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

elif simu in ['DFLAT','DF500','DTRI2','DMO10','DMOPL']:
    seg = 'M100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:04d}.nc'.format(i) for i in range(6,109,6)]
    # seg = 'M100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:04d}.nc'.format(i) for i in range(84,85,6)] # only day
    
elif simu=='E600M':
    # seg = '19AUG'; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:04d}.nc'.format(i) for i in range(6,7,1)]
    seg = '19AU2'; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:04d}.nc'.format(i) for i in range(1,38,6)]
    
elif simu=='E620M':
    days = [19,20]
    hours = [10,14,18]
    # hours = [0,4,8,12,16,20]
    tree = 2
    seg = ['AROME','600m','200m'][tree]
    lFiles=[]
    for day in days:
        if tree==0:
            lFiles += [dataPath + '{:02d}AUG_{:02d}H.OUT.nc'.format(day,i) for i in hours]
        else:
            lFiles += [dataPath + simu+'.'+str(tree)+'.{:02d}AUG.OUT.{:04d}.nc'.format(day,i) for i in hours]
            
f0 = xr.open_dataset(lFiles[0],engine='h5netcdf')
var_names = [i for i in f0.data_vars]
        
nx1 = 1 ; nx2 = len(f0.ni)-1
ny1 = 1 ; ny2 = len(f0.nj)-1
nz1 = 1 ; nz2 = len(f0.level)-1

x = np.round( np.array(f0.ni)[nx1:nx2] ,decimals=-1)
y = np.round( np.array(f0.nj)[ny1:ny2] ,decimals=-1)
z = np.array(f0.level)[nz1:nz2]
z_ = np.array(f0.level_w)[nz1:nz2]
dz = z_[1:]-z_[:-1]
dz = np.append(dz,dz[-1])

Zm = np.array(f0.level)[nz1:]
ZTOP = (Zm[-1]+Zm[-2])/2


if simu[0]=='A':
    z_max_skyview = 16001
elif simu[0]=='C':
    z_max_skyview = 4001
elif simu[0]=='D':
    z_max_skyview = 5001
elif simu[0]=='E':
    z_max_skyview = 15001
else:
    z_max_skyview = 15001
    
    
dx = x[1]-x[0]
nt,nz,ny,nx = len(lFiles),len(z),len(y),len(x)


pgdFile = None
if simu[0] in ['A','C','D']:
    pgdFile = dataPath+simu+'_init_'+seg+'_pgd.nc'
else:
    pgdFile = dataPath+'PGD_'+simu+'_'+str(round(dx))+'m.nc'
    # pgdFile = dataPath+'PGD_200m.neste1.nc'
    
if pgdFile is not None:
    ZS = xr.open_dataset(pgdFile)['ZS'][ny1:ny2,nx1:nx2].data
else:
    ZS = np.zeros((ny,nx))
    
if orog:
    ZSmean = np.mean(ZS)
    MO = ZS>ZSmean # Mountain region (upper one)
    PL = np.logical_not(MO) # Plain region (lower one)
    
# Z = np.copy(z)
# Z = np.arange(25.,z[-1],50.)
Z = np.arange(25.,6000,50.)
nZ = len(Z)
dZ = 50*np.ones(nZ)

dz_shrink = (ZTOP-ZS) /ZTOP

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

def emptyness(arrayCOO,end='\n'):
    print('Volume = {:02%}'.format(arrayCOO.nnz/arrayCOO.size),end=end)

def make_sparses_skyview(rcloud,rprecip,w,thetav,savePath):
    if not os.path.exists(savePath): os.makedirs(savePath) ; print('Directory created !')

    dtype = np.float32
    idx_dtype = np.int32

    new_z = np.arange(0.,z_max_skyview,dx,dtype=dtype)
    end = '--- '
    
    print("Interpolation",end=end)
    new_rcloud = AL_lin_interp(rcloud,new_z,Zm,ZS)
    new_rprecip = AL_lin_interp(rprecip,new_z,Zm,ZS)
    # new_rtracer = AL_lin_interp(rtracer,new_z,Zm,ZS)
    new_w =AL_lin_interp(w[1:],new_z,Zm[1:]/2+Zm[:-1]/2,ZS)
    new_thva = AL_lin_interp( thetav,new_z,Zm,ZS)
    new_thva -= np.nanmean(new_thva,axis=(1,2),keepdims=True)
    
    print("Threshold",end=end)
    new_rcloud[new_rcloud < 1e-6 ] = np.nan
    new_rprecip[new_rprecip < 1e-4 ] = np.nan
    new_w[np.abs(new_w) < 2.] = np.nan
    new_thva[20:,:,:] = np.nan # 11 for BOMEX # 14 for AMMA # 8 for LBA # 20 for AMOPL
    new_thva[new_thva > -1] = np.nan
    
    print("Sparse",end=end)
    new_rcloud_sparse = sparse.COO.from_numpy(new_rcloud.reshape((1,)+new_rcloud.shape),idx_dtype=idx_dtype,fill_value=np.nan)
    new_rprecip_sparse = sparse.COO.from_numpy(new_rprecip.reshape((1,)+new_rprecip.shape),idx_dtype=idx_dtype,fill_value=np.nan)
    # new_rtracer_sparse = sparse.COO.from_numpy(new_rtracer.reshape((1,)+new_rtracer.shape),idx_dtype=idx_dtype,fill_value=np.nan)
    new_w_sparse = sparse.COO.from_numpy(new_w.reshape((1,)+new_w.shape),idx_dtype=idx_dtype,fill_value=np.nan)
    new_thva_sparse = sparse.COO.from_numpy(new_thva.reshape((1,)+new_thva.shape),idx_dtype=idx_dtype,fill_value=np.nan)
    
    sparse.save_npz(savePath+'rcloud_sparse_'+simu+'_'+str(it) ,new_rcloud_sparse)  
    sparse.save_npz(savePath+'rprecip_sparse_'+simu+'_'+str(it) ,new_rprecip_sparse)
    # sparse.save_npz(savePath+'rtracer_sparse_'+simu+'_'+str(it) ,new_rtracer_sparse)
    sparse.save_npz(savePath+'w_sparse_'+simu+'_'+str(it) ,new_w_sparse)
    sparse.save_npz(savePath+'thva_sparse_'+simu+'_'+str(it) ,new_thva_sparse)

    print("Done",end=end)
    emptyness(new_rcloud_sparse,end=end)
    emptyness(new_rprecip_sparse,end=end)
    # emptyness(new_rtracer_sparse,end=end)
    emptyness(new_w_sparse,end=end)
    emptyness(new_thva_sparse)
    
#%%
k = 2*np.pi*np.fft.fftfreq(nx,dx)
m = 2*np.pi*np.fft.fftfreq(ny,dx)
kk,mm = np.meshgrid(k,m)
k2 = kk**2+mm**2
del(kk,mm)
def wind_filter(U,V,lc=5e3):
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
if make_maps:
    Nmaps=3 if '3maps' in img_maps_names[0] else 1
    ft=40
    wspace = 0.
    if Nmaps==3:
        fig_maps_BL,axs_maps_BL=plt.subplots(ncols=Nmaps,figsize=(32,18))#(16,9))#,constrained_layout=True)
        plt.subplots_adjust(left=0.0+wspace,right=1-wspace,bottom=0.35, top=0.94,wspace=wspace,hspace=wspace)
        text_title_BL = fig_maps_BL.text(0.5,0.97,simu+' '+seg+'  '+'00 h 00',fontsize=1.5*ft,ha='center',va='center')
        
        fig_maps_SL,axs_maps_SL=plt.subplots(ncols=Nmaps,figsize=(32,18))#(16,9))#,constrained_layout=True)
        plt.subplots_adjust(left=0.0+wspace,right=1-wspace,bottom=0.35, top=0.94,wspace=wspace,hspace=wspace)
        text_title_SL = fig_maps_SL.text(0.5,0.97,simu+' '+seg+'  '+'00 h 00',fontsize=1.5*ft,ha='center',va='center')
        
    if Nmaps==1:
        fig_maps_list,axs_maps_list,text_title_list = [None]*len(img_maps_names) , [None]*len(img_maps_names) , [None]*len(img_maps_names)
        for i in range(len(img_maps_names)):
            fig_maps_list[i],axs_maps_list[i]=plt.subplots(ncols=Nmaps,figsize=(10,10))#(16,9))#,constrained_layout=True)
            plt.subplots_adjust(left=0.0+wspace,right=1-wspace,bottom=0., top=1.,wspace=wspace,hspace=wspace)
            text_title_list[i] = fig_maps_list[i].text(0.5,0.97,simu+' '+seg+'  '+'00 h 00',fontsize=1.0*ft,ha='center',va='center')
            
    nstarts = 40
    px,py = np.meshgrid(x[::nx//nstarts]/1000,y[::ny//nstarts]/1000)
    start_points = np.transpose(np.array([px.flatten(),py.flatten()]))

def is_thermal_TR_W(TR,W):
    TR_std = np.std(TR,axis=(1,2))
    TR_mean = np.mean(TR,axis=(1,2))
    TR_stdmin = 0.5/(z+dz/2) * np.cumsum(dz*TR_std) # 0.05
    TR_threshold = TR_mean+np.maximum(TR_std,TR_stdmin)
    return np.logical_and( TR>TR_threshold.reshape(nz,1,1) , W>0 )

def is_thermal_TR_W_THVBL(TR,W,THV,THVBL):
    TR_std = np.std(TR,axis=(1,2))
    TR_mean = np.mean(TR,axis=(1,2))
    # TR_stdmin = 0.5/(z+dz/2) * np.cumsum(dz*TR_std) # 0.05
    TR_stdmin = 0.
    TR_threshold = TR_mean+np.maximum(TR_std,TR_stdmin)
    return np.logical_and( np.logical_and( TR>TR_threshold.reshape(nz,1,1) , W>0)  , THV< THVBL.reshape((1,ny,nx))+0.5 )

def is_thermal_TR_W_agl(TR,W,ZS,z):
    agl = z.reshape(nz,1,1) * dz_shrink.reshape(1,ny,nx)
    groundTR = np.mean(TR[0])# + np.std(TR[0])
    return np.logical_and( TR> groundTR/(10-9*np.exp(-agl/200))*np.exp(-agl/2000), W>0)

from identification_methods import identify
listVarNames = ['SVT001','WT']#[]#
def thermalMask(dictVars):
    TR = dictVars['SVT001'][0,nz1:nz2,ny1:ny2,nx1:nx2]
    W = dictVars['WT'][0,nz1:nz2,ny1:ny2,nx1:nx2]
    mask = np.zeros((1,nz+2,ny+2,nx+2),dtype='bool')
    mask[0,nz1:nz2,ny1:ny2,nx1:nx2] = is_thermal_TR_W(TR,W)
    mask[0,:,0,:] = mask[0,:,-2,:]
    mask[0,:,-1,:] = mask[0,:,1,:]
    mask[0,:,:,0] = mask[0,:,:,-2]
    mask[0,:,:,-1] = mask[0,:,:,1]
    return mask
    
def plot_maps(fig_maps,axs_maps,text_title,ZS,maps_VAR,maps_params, PRS,MASK ,U,V,legend=False,short_title=False,ZS_labels=True,map_name=''):
    
    flat_precip = get_flat_precip(x,y,flat_X,flat_Y,PRS)
    select_precip = flat_precip > 10**-4 # 1 mm of rain in 1 min
    cb_length = 0.9 /Nmaps
    cb_width = 0.03
    cb_posy = 0.20
    shrink = 1.
    
    for i in range(Nmaps):
        ax = axs_maps[i] if Nmaps > 1 else axs_maps
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim([x[0]/1000,x[-1]/1000])
        ax.set_ylim([y[0]/1000,y[-1]/1000])
        # ax.set_xticks([])
        if i>=0: ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=ft*2/3)
        ax.set_xlabel('km',fontsize=ft,rotation=0)
        if orog:
            # CS = ax.contour(x/1000,y/1000,ZS, levels=[np.percentile(ZS,5),np.mean(ZS),np.percentile(ZS,95)],linewidths=1.,alpha=0.5,colors=['k'] , zorder=2)
            CS = ax.contour(x/1000,y/1000,ZS, levels=[0,1000,2000,3000,4000],linewidths=1.,alpha=0.5,colors=['k'] , zorder=2)
            if ZS_labels: plt.clabel(CS, CS.levels, inline=True, inline_spacing=0, fmt='%d', fontsize=ft/3)
        
        cf = ax.pcolormesh(x/1000,y/1000,maps_VAR[i],vmin=maps_params[i][0], vmax=maps_params[i][1],cmap=maps_params[i][2],shading='gouraud')
        if legend:
            cax = fig_maps.add_axes([(2*i+1)/Nmaps/2-cb_length/2, cb_posy, cb_length, cb_width])
            cb = plt.colorbar(cf,cax=cax,shrink=shrink,orientation = "horizontal",extend=maps_params[i][3],format=maps_params[i][4])
            if maps_params[i][5] is not None:
                cb.set_ticks(maps_params[i][5]) # [-5,-2,-1,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1.,2.,5.]
            cb.ax.tick_params(labelsize=ft*2/3)
            cb.ax.set_title(maps_params[i][6],fontsize=ft)
            cb.set_label(maps_params[i][7],fontsize=ft,rotation=0)
        
        if maps_params[i][8]:
            ax.scatter(flat_X[select_precip]/1000, flat_Y[select_precip]/1000, s= flat_precip[select_precip]*1e2, c='black',zorder=100)
        
        # ax.contour(x/1000,y/1000,CBM, levels=[0.5],linewidths=2,alpha=1,colors=['grey'])#,colors=[CTcm( (z[cloud_top[obj]]/1000 -ct_lev[0])/ct_lev[-1] )])
        if maps_params[i][9]=='CBM':
            ax.contour(x/1000,y/1000,MASK, levels=[0.5],linewidths=1,alpha=1,colors=['grey'] )
        if maps_params[i][9] in ['TBM','TTM']:
            ax.contour(x/1000,y/1000,MASK, levels=[0.5],linewidths=0.5,alpha=0.7,colors=['darkgreen'] )
            # ax.contourf(x/1000,y/1000,CBM,alpha=0.5,hatches=[None, '//'])
            # ax.contourf(x/1000,y/1000,CBM,2,colors=['none','k'],alpha=0.5)
        
        if maps_params[i][10] is not None:
            Umap,Vmap = wind_filter(U,V,maps_params[i][10])
            ax.streamplot(x/1000,y/1000, Umap,Vmap, density=3, color='k', linewidth=3/10*np.sqrt(Umap**2+Vmap**2), arrowsize = 0.5,maxlength=1.,minlength=0.)#
            
            # ax.streamplot(x/1000,y/1000, Umap,Vmap, density=20, color='k', linewidth=3/10*np.sqrt(Umap**2+Vmap**2), arrowsize = 0. , start_points=start_points,maxlength=1.,minlength=0.1)#
            # streamp = 
            # for il,arrows in enumerate(streamp.arrows):
            #     if max(streamp.lines[il].linewidths) < 1: arrows[il].remove()
    if legend:
        if maps_params[i][8]:
            cax=fig_maps.add_axes([2/Nmaps-0.1,0.08, 0.2, 0.03])
            cax.set_xlim([0.5,4.5])
            cax.set_ylim([0.,2.])
            cax.set_yticks([])
            cax.set_xticks([1,2,3,4])
            cax.set_xticklabels(['0.1','1','10','100'],size=ft*2/3)
            # cax.xticks.label.set_fontsize(2/3*ft) 
            cax.scatter([1,2,3,4],[1,1,1,1],s=0.5*np.array([0.1,1,10,100]), c='black')
            cax.set_xlabel('mm / min',fontsize=ft)
            cax.set_title('Rain at the surface',fontsize=ft)
        if maps_params[i][10] is not None:
            cax=fig_maps.add_axes([1/Nmaps-0.1,0.08, 0.2, 0.03])
            cax.set_xlim([0.5,4.5])
            cax.set_ylim([0.,2.])
            cax.set_yticks([])
            cax.set_xticks([1,2,3,4])
            cax.set_xticklabels(['1','5','10','20',],size=ft*2/3)
            US,VS = np.array([[1,5,10,20],[1,5,10,20],[1,5,10,20]]) , np.zeros((3,4))
            cax.streamplot(np.arange(1,5),np.arange(3),US,VS, density=3, color='k', linewidth=3/10*np.sqrt(US**2+VS**2) , arrowsize = 0.5,start_points=np.array([[2.5,1.]]) )
            cax.set_xlabel('m/s',fontsize=ft)
            cax.set_title('Wind speed',fontsize=ft)
        
    date = str(f.time.data[0])[:10]
    hour = '{:02d} h {:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute))
    if short_title:
        text_title.set_text(date+'  '+hour if dates else hour)
    else:
        text_title.set_text(map_name+' ' +simu+' '+seg+'  '+(date+'  '+hour if dates else hour))
        
        
    if not os.path.exists(savePathFigures+'maps_video/'): os.makedirs(savePathFigures+'maps_video/') ; print('Directory created !')
    name = savePathFigures+'maps_video/maps_video_'+map_name+'_'+simu+'_'+seg+'_'+ ( date+'_'+hour if dates else hour) +'.png'
    fig_maps.savefig(name)
    # print(' ---- Saved '+date+'  '+hour)
    return name
    
if make_hist2D:
    nvar = 15
    nrows=3
    ncols=(nvar +1)//nrows
    fig_hist2D,axs_hist2D = plt.subplots(nrows=nrows,ncols=ncols,figsize=(19.2,10.8))
    wspace = 0.1/nvar
    plt.subplots_adjust(left=0.04,right=0.92,bottom=0.07,top=0.92,wspace=wspace,hspace=0.22)
    text_title_hist2D = fig_hist2D.text(0.5,0.97,simu+' '+seg+' distribution profiles at 00 h 00',fontsize=20,ha='center',va='center')
    # plt.suptitle(,fontsize=ft)
    
    
def plot_hist2D(z,W,TH,TKE,surflay,thermals,ft=15): #RV, RC , RP , RT
    nz = len(z)
    ## Cas avec de l'eau
    # anom_water = RT - np.nanmean(RT,axis=(1,2),keepdims=True)
    # flux_water = w*anom_water
    # T = TH * (P/100000.)**(2/7)
    # iso = z[np.argmin(np.abs(np.mean(T,axis=(1,2))-T0))]
    # RH = RH_water_ice(rv,T,P)
    # thetav = TH * (1+ 1.61*RV)/(1+water)
    # thetali = TH - TH/T /Cp * ( Lv(T)*(RC+RR) + Ls(T)*(RI+RS+RG) )
    
    ## Sans eau
    thetav = TH
    thetali = TH
    
    anom_thetav = thetav - np.nanmean(thetav,axis=(1,2),keepdims=True)
    anom_thetali = thetali - np.nanmean(thetali,axis=(1,2),keepdims=True)
    flux_thetali = W*anom_thetali
    
    ## Présentation convection profonde
    # nvar = 18
    # nrows=2
    # pdf_variables = [ 'THT', water, 'RVT', 'RCT', 'RRT', anom_thetav , 'WT' , flux_water , flux_thetali,
    #                   'UT' , 'VT' , 'RIT', 'RST', 'RGT', anom_thetav , 'WT' , flux_water , flux_thetali ]
    # masks = [None,RC>1e-6]
    # pdf_mask = [0,0,0,0,0,0,0,0,0,
    #             0,0,0,0,0,1,1,1,1]
    # pdf_names = ['$\\theta$','$r_t$','$r_v$','$r_l$','$r_r$',  '$ \\theta _v^{\prime} $', '$w$','$w·r_t^{\prime}$','$w·\\theta _{li}^{\prime}$',
    #              '$u$','$v$','$ r_i$', '$r_s$','$r_g $', '$ (\\theta _v^{\prime})^{cloud}$', '$w^{cloud}$','$(w·r_t^{\prime})^{cloud}$','$(w·\\theta _{li}^{\prime})^{cloud}$']# '$-\\frac{\\partial u}{\\partial x}-\\frac{\\partial v}{\\partial y}$', 
    # pdf_units = [' K  ', 'kg/kg','kg/kg','  kg/kg ','  kg/kg ',' K ', ' m/s ',' m/s·kg/kg  ',' m/s·K ',
    #              ' m/s ',' m/s ','kg/kg','  kg/kg ','  kg/kg ',' K ', ' m/s ',' m/s·kg/kg  ',' m/s·K ']#' \frac{m/s}{km} ', 
    # wm = 23 # m/s # wind maximum value
    # rmax=-1.6
    # rmin=-5.5
    # pdf_ranges = [ [290,359,'lin'],[0,0.018,'lin'],[rmin,rmax,'log'],[rmin,rmax,'log'],[rmin,rmax,'log'] ,[-6,6,'lin'], [-wm,wm,'lin'], [-0.35,0.35,'lin'], [-350,350,'lin'],
    #               [-wm,wm,'lin'],[-wm,wm,'lin'],[rmin,rmax,'log'],[rmin,rmax,'log'],[rmin,rmax,'log'] ,[-6,6,'lin'], [-wm,wm,'lin'], [-0.35,0.35,'lin'], [-350,350,'lin']] #,[-35,35,'lin']
    
    ## Présentation convection peu profonde sèche
    # nvar = 15
    # nrows=3
    pdf_variables = [ TH, anom_thetav , W , flux_thetali, TKE, 
                      TH , anom_thetav , W , flux_thetali, TKE, 
                      TH , anom_thetav , W , flux_thetali, TKE ]
    masks = [None,surflay,thermals] # RC>1e-6
    # mask_names = ['all','t','s']
    pdf_mask = [0,0,0,0,0,
                1,1,1,1,1,
                2,2,2,2,2]
    # pdf_names = ['$\\theta ^{all}$','$ { \\theta ^{\prime} }^{all}$', '$w^{all} $','${ w·\\theta^{\prime} }^{all}$','$k ^{all}$',
    #              '$u ^{all}$'      ,'$ { \\theta ^{\prime} } ^s $', '$w^s $','${ w·\\theta ^{\prime} } ^s$','$k ^s$',
    #              '$v ^{all}$'      ,'$ { \\theta ^{\prime} } ^t $', '$w^t $','${ w·\\theta ^{\prime} } ^t$','$k ^t$' ]
    pdf_names = ['$\\theta ^{all}$','$ { \\theta ^{\prime} }^{all}$', '$w^{all} $','${ w·\\theta^{\prime} }^{all}$','$k ^{all}$',
                 '$\\theta ^{s}$'      ,'$ { \\theta ^{\prime} } ^s $', '$w^s $','${ w·\\theta ^{\prime} } ^s$','$k ^s$',
                 '$\\theta ^{t}$'      ,'$ { \\theta ^{\prime} } ^t $', '$w^t $','${ w·\\theta ^{\prime} } ^t$','$k ^t$' ]
    pdf_units = [' K  ' ,' K ', ' m/s ',' m/s·K ',' $m^2 / s^2$',
                 ' m/s ',' K ', ' m/s ',' m/s·K ',' $m^2 / s^2$',
                 ' m/s ',' K ', ' m/s ',' m/s·K ',' $m^2 / s^2$' ]
    wm = 7 # m/s # wind maximum value
    tam = 6 # K #  max temperature anomaly
    thmax = 330
    thmin = 300
    wtam = 8
    tkem = 3. # wm**2/2
    pdf_ranges = [[thmin,thmax,'lin'],[-tam,tam,'lin'], [-wm,wm,'lin'], [-wtam,wtam,'lin'], [0,tkem,'lin'],
                  [thmin,thmax,'lin'],[-tam,tam,'lin'], [-wm,wm,'lin'], [-wtam,wtam,'lin'], [0,tkem,'lin'],
                  [thmin,thmax,'lin'],[-tam,tam,'lin'], [-wm,wm,'lin'], [-wtam,wtam,'lin'], [0,tkem,'lin'] ]
    
    nbins = 100
    nz_max = nz
    if nz_max==nz:
        z_bins = np.append(z[:-1] + (z[1:] - z[:-1])/2 , z[-1]+(z[-1] - z[-2])/2)
    else:
        z_bins = z[:nz_max] + (z[1:nz_max+1] - z[:nz_max])/2
    
    
    for iv in range(len(pdf_variables)):
        ax = axs_hist2D[iv//ncols,iv%ncols]
        ax.clear()
        if pdf_ranges[iv][2] == 'lin':
            bins = np.linspace(pdf_ranges[iv][0],pdf_ranges[iv][1],nbins)
    #        ax.set_xlim([pdf_ranges[iv][0],pdf_ranges[iv][1]])
        elif pdf_ranges[iv][2] == 'log':
            bins = np.insert( np.logspace(pdf_ranges[iv][0],pdf_ranges[iv][1],nbins-1) , 0, 0., axis=0)
            ax.set_xscale('log')
            ax.set_xlim([10**pdf_ranges[iv][0],10**pdf_ranges[iv][1]])
            
        var = pdf_variables[iv]
        hist_field = np.zeros((nz_max-1,nbins-1))
        for iz in range(1,nz_max) :
            mask = masks[pdf_mask[iv]]
            if mask is None:
                hist_field[iz-1,:], _ = np.histogram( var[iz] ,bins=bins, density=False)
            else:
                hist_field[iz-1,:], _ = np.histogram( var[iz][mask[iz]] ,bins=bins, density=False)
            
        ax_X, ax_Y = np.meshgrid(  bins, z_bins/1000 )
        im = ax.pcolormesh(ax_X,ax_Y,hist_field/nx/ny, norm=matplotlib.colors.LogNorm(vmin=1/nx/ny,vmax=1.), shading= 'flat',cmap='magma_r')#, cmap='gnuplot2_r')## cmap='hot_r')#,cmap='nipy_spectral_r')#
        ax.set_title(pdf_names[iv] + '   ('+pdf_units[iv]+')' ,fontsize=ft)
        
        
        if pdf_ranges[iv][0] == - pdf_ranges[iv][1]:
            ax.axvline(0,linestyle='--',color='k',lw=1)
        ax.plot( np.sum(hist_field * (bins[1:]/2+bins[:-1]/2).reshape((1,nbins-1)) ,axis=1 )/np.sum(hist_field,axis=1) ,(z_bins[1:]+z_bins[:-1])/2000  , color='chartreuse',lw=1.5)
        
        # if pdf_variables[iv] in ['RCT', 'RRT','RIT', 'RST', 'RGT']:
        #     ax.axhline(iso/1000,linestyle='--',color='b',lw=1.5)
        #     ax.text(bins[2],iso/1000,'0°C',fontsize=ft,color='b', ha="left", va="center")

        if iv==len(pdf_variables)-1:# and file == lFiles[0]:
            pdf_cb = plt.colorbar(im, cax=fig_hist2D.add_axes([0.93, 0.2, 0.02, 0.6]) ,ticks=[10**-5,10**-4,10**-3,10**-2,10**-1,1],shrink=1.,orientation = "vertical")
            pdf_cb.ax.tick_params(labelsize=ft*2/3)
            pdf_cb.set_label("Density",fontsize=ft)
        
        ax.set_ylim([-0.1,z_bins[-1]/1000+0.1])
        if iv%ncols==0:
            ax.set_ylabel('Altitude (km)',fontsize=ft*2/3)
        else:
            # ax.set_yticks([])
            ax.set_yticklabels([])
#        ax.set_xticks([])
        # ax.set_xlabel(pdf_units[iv],fontsize=ft*2/3)
        ax.grid(axis='y')
        
    date = str(f.time.data[0])[:10]
    hour = '{:02d} h {:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute))
    # plt.suptitle(simu+' '+seg+' distribution profiles at '+(date+'  '+hour if dates else hour),fontsize=ft)
    text_title_hist2D.set_text(simu+' '+seg+' distribution profiles at '+(date+'  '+hour if dates else hour))
                                        
    if not os.path.exists(savePathFigures+'pdf_profiles/'): os.makedirs(savePathFigures+'pdf_profiles/') ; print('Directory created !')
    name = savePathFigures+'pdf_profiles/pdf_profiles_'+simu+'_'+seg+'_'+ ( date+'_'+hour if dates else hour) +'.png'
    fig_hist2D.savefig(name)
    # print(' ---- Saved '+date+'  '+hour)
    return name
    
#%%
if make_hori_mean:
    VAR3D_2mean_names = ['ALPHA','RHO','TH','RV','THV','CF','P','TR','W','U','V','TKE','WTH']
    VAR3D_2mean_units = ['','kg/m3','K','kg/kg','K','','Pa','kg/kg','m/s','m/s','m/s','m²/s²','m/sK']
    VAR2D_2mean_names = ['H','LE','TSE','TPW','PRS','BLH','THVBL']
    VAR2D_2mean_units = ['W/m²','W/m²','J/m²','kg/m²','mm/s','m','K']
    nvar2D = 7
    nvar3D = 13 # 14
    nflux = 4
    
    regions = [] ; region_types = [] ; region_names = []
    regions.append(np.ones((ny,nx),dtype=np.bool_))    ; region_types.append('2D') ; region_names.append('a')
    regions.append(np.ones((ny,nx),dtype=np.bool_))    ; region_types.append('3D') ; region_names.append('s')
    regions.append(np.ones((nZ,ny,nx),dtype=np.bool_)) ; region_types.append('3D') ; region_names.append('t')
    regions.append(np.ones((nZ,ny,nx),dtype=np.bool_)) ; region_types.append('3D') ; region_names.append('e')
    if orog:
        regions.append(MO) ; region_types.append('2D') ; region_names.append('MO')
        regions.append(PL) ; region_types.append('2D') ; region_names.append('PL')
        dzrange = 250
        for zrange in range(0,int(ZS.max()),dzrange):
            reg3D = np.zeros((nZ,ny,nx),dtype=np.bool_)
            reg2D = np.logical_and( ZS>= zrange , ZS<zrange+dzrange )
            reg3D[:,reg2D] = True
            regions.append(reg3D) ; region_types.append('3D') ; region_names.append('t_'+str(zrange))
    
    nreg = len(regions)         
    # timeH = np.zeros(nt)
    
    data3D_mean = np.full((nreg,nvar3D,nt,nZ),np.nan,dtype=floatype)
    data2D_mean = np.zeros((nreg,nvar2D,nt),dtype=floatype)
    if orog: data_flux_MO = np.full((nflux,nt,nz),np.nan,dtype=floatype)

#%%
times = f0.time.data
it0 = 1

img_maps_list = [[] for i in range(len(img_maps_names))]

img_hist2D = []
for it,fname in enumerate(lFiles):
    print()
    time0 = time.time()
    f = xr.open_dataset(fname,engine='h5netcdf')
    def npf(v,dtype=floatype):
        if v in var_names:
            return np.array(f[v][0,nz1:nz2,ny1:ny2,nx1:nx2],dtype=dtype)
        else:
            return np.zeros((nz2-nz1,ny,nx),dtype=dtype)
    def npf2D(v,dtype=floatype):
        if v in var_names:
            return np.array(f[v][0,ny1:ny2,nx1:nx2],dtype=dtype)
        else:
            return np.zeros((ny,nx),dtype=dtype)
        
    if it>0: times = np.concatenate((times,f.time.data))
    timeH = ' {:02d}h{:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute)) # date = str(f.time.data[0])[:10]+
    # timeH[it] = f.time.dt.hour + f.time.dt.minute/60
    print(time.ctime()[-13:-5], timeH ,end=' ')
    
    U = npf('UT') ; U = (U+np.roll(U,1,axis=2))/2
    V = npf('VT') ; V = (V+np.roll(V,1,axis=1))/2
    W = npf('WT') ; W = (W+np.roll(W,-1,axis=0))/2
    TKE = npf('TKET')
    P = npf('PABST')
    TH = npf('THT')
    RV = npf('RVT')
    RC = npf('RCT')+npf('RIT')
    RP = npf('RRT')+npf('RST')+npf('RGT')
    RT = RV+RC+RP
    TR = npf('SVT001')
    
    PRS = npf2D('ACPRRSTEP')

    print(' Read :',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
    # T = TH * (P/p0)**(2/7)
    THV = TH * (1.+ Md/Mv*RV)/(1.+ RT )
    RHO = P/Rd/ (THV * (P/p0)**(2/7))
    H = npf2D('DUMMY2D1') * Cp * RHO[0]  # sensible heat flux K*m/s  to   W/m²   mutliplying by rho and     Cp = 1004.71 J/K/kg
    LE = npf2D('DUMMY2D2') * Lv * RHO[0]
    CONV = horizontal_convergence(U,V,dx)
    
    thermals = np.zeros((nz2-nz1,ny,nx),dtype=np.int32)
    BLH,CBM,TTM,THVBL,WBL,UBL,VBL,UBLH,VBLH = boundary_layer_diags(Zm,ZS,RC,THV,RHO,W,U,V,thermals)
    SLD,TBM,THVSL,WSL,USL,VSL = surface_layer_diags(Zm,ZS,THV,RHO,W,U,V,thermals)
    
    if 'SVT001' in var_names and make_charac:
        thermals,_ = identify(fname, listVarNames, thermalMask, name="tracer_updraft",delete=10,write=False,rename=False)
        thermals = np.array(thermals[0,nz1:nz2,ny1:ny2,nx1:nx2],dtype=np.int32)
        change_label(thermals)
        print(' Thermals:',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    elif 'SVT001' in var_names:
        # thermals = is_thermal_TR_W(TR,W)
        # thermals = is_thermal_TR_W_THVBL(TR,W,THV,THVBL)
        thermals = is_thermal_TR_W_agl(TR,W,ZS,z)
    # else:
    #     thermals = np.zeros((nz2-nz1,ny,nx),dtype=np.int32)
        
    SLD = 20. * np.ones_like(SLD)
    SL = AL_SL(Z,ZS,SLD)

    TSE = dz_shrink * np.sum( dz.reshape(nz,1,1)*RHO*Cp*TH , axis=0) # total sensible energy # J/m^2
    TPW = dz_shrink * np.sum( dz.reshape(nz,1,1)*RHO*RT , axis=0) # total precipitable water # kg/m^2
    
    print(' Compute variables:',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
    
    
    if make_charac:
        # variables = np.array([U,V,W,CONV,THV,P,TR,RC,RP,RT,W*RHO*RT],dtype=np.float32 ) 
        variables = [U,V,W,CONV,THV,P,TR,RC,RP,RT,W*RHO*RT]
        charac,area,pos = charac_opti_3D(thermals, variables )
        print(' Charac:',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
        if not os.path.exists(savePathObjects): os.makedirs(savePathObjects) ; print('savePath directory created !')
        np.savez_compressed(savePathObjects+'objects_charac_area_pos.{:03d}'.format(it+it0),objects=thermals,charac=charac,area=area,pos=pos)
        print(' Save charac:',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
        
    if make_sparses:
        make_sparses_skyview(RC,RP,W,THV,savePathSkyView)
        print(' SkyView sparses:',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
    if make_hori_mean:
        # if orog:
        RHO_ = AL_lin_interp( RHO,Z,Zm,ZS)
        TH_ = AL_lin_interp( TH,Z,Zm,ZS)
        RV_ = AL_lin_interp(RV,Z,Zm,ZS)
        THV_ = AL_lin_interp( THV,Z,Zm,ZS)
        RC_ = AL_lin_interp(RC,Z,Zm,ZS)
        P_ = AL_lin_interp( P,Z,Zm,ZS)
        TR_ = AL_lin_interp(TR,Z,Zm,ZS)
        W_ = AL_lin_interp(W,Z,Zm,ZS)
        U_ = AL_lin_interp(U,Z,Zm,ZS)
        V_ = AL_lin_interp(V,Z,Zm,ZS)
        TKE_ = AL_lin_interp(TKE,Z,Zm,ZS)
        thermals_ = AL_objects(thermals,Z,Zm,ZS) ; thermals_[W_<=0] = 0
        # else:
        #     RHO_=RHO ; TH_=TH ; RV_=RV ; THV_=THV ; RC_=RC ; P_=P ; TR_=TR ; W_=W ; U_=U ; V_=V ; TKE_=TKE ; thermals_ = thermals
        print('Interpolations on altitude levels :',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
        
        regions[1] = SL
        regions[2] = np.logical_and( thermals_>0  , ~SL )
        regions[3] = np.logical_and( thermals_==0 , ~SL )
        
        VAR3D_2mean = ['ALPHA',RHO_,TH_,RV_,THV_,RC_>1e-6,P_,TR_,W_,U_,V_,TKE_,'WTH']
        VAR2D_2mean = [H,LE,TSE,TPW,PRS,BLH,THVBL]
        for ir,reg in enumerate(regions):
            for iv,VAR in enumerate(VAR3D_2mean):
                if region_types[ir] == '2D':
                    if type(VAR)==str:
                        if VAR=='ALPHA':
                            data3D_mean[ir,iv,it,:] = np.mean(RHO_[:,reg]>0,axis=1)
                        elif VAR=='WTH':
                            WpTHp = ( W_-np.nanmean(W_[:,reg],axis=1).reshape(nZ,1,1)) * (TH_-np.nanmean(TH_[:,reg],axis=1).reshape(nZ,1,1))
                            data3D_mean[ir,iv,it,:] = np.nanmean(WpTHp[:,reg],axis=1)
                    else:
                        data3D_mean[ir,iv,it,:] = np.nanmean(VAR[:,reg],axis=1)
                    
                elif region_types[ir] == '3D':
                    if region_names[ir][:2]=='t_':
                        reg = np.logical_and(reg,regions[2])
                    for iz in range(nZ):
                        if type(VAR)==str:
                            if VAR=='ALPHA':
                                data3D_mean[ir,iv,it,iz] = np.mean(reg[iz]>0)
                            elif VAR=='WTH':
                                data3D_mean[ir,iv,it,iz] = np.mean( ( W_[iz,reg[iz]]-np.mean(W_[iz,reg[iz]])) * ( TH_[iz,reg[iz]]-np.mean(TH_[iz,reg[iz]])) )
                        else:
                            data3D_mean[ir,iv,it,iz] = np.nanmean(VAR[iz,reg[iz]])
            
            for iv,VAR in enumerate(VAR2D_2mean):
                if region_types[ir] == '2D':
                    data2D_mean[ir,iv,it] = np.mean(VAR[reg])
        
        if orog:
            data_flux_MO[:,it,:] = MO_flux(MO,U,V,RHO,RV,RT,TH)
        
        print('Nanmeans :',str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
    
    if make_maps:
        # THVAS = AGL_lin_anomaly(THV[0],2,Z,data3D_mean[2,it],ZS) ##  [-4,4,'RdBu_r','both','%.1f',None,"$\\theta _v ^{\prime}$ and wind at 5m AGL",' K',True,None,'10m']
        
        for i,maps_name in enumerate(img_maps_names):

            if maps_name=='video_3maps_BL':
                maps_VAR = [BLH,THVBL,WBL]
                maps_params = [[0.,5000.,'rainbow','max','%d',[0,1000,2000,3000,4000],"Top of the mixed layer",' m',True,None,None],
                               [300,320,'turbo','both','%d',None,"$\\theta_v$ and mean wind",' K',True,None,2e3],
                               # [292.5,322.5,'turbo','both','%d',None,"$\\theta_v$ and wind at 10m AGL",' K',True,None,'10m'],
                                [-5,5,Wcm,'both','%.1f',None,"Mean W and thermal top",' m/s',True,'TTM',None] ]
                img_maps_list[i].append( plot_maps(fig_maps_BL,axs_maps_BL,text_title_BL,ZS,maps_VAR,maps_params, PRS,TTM,UBL,VBL,legend=True if it == 0 else False ,map_name='BL') )
            elif maps_name=='video_3maps_SL':
                maps_VAR = [SLD,THVSL,WSL]
                maps_params = [[0.,100.,'rainbow','max','%d',None,"Depth of the surface layer",' m',True,None,None],
                               [300,320,'turbo','both','%d',None,"Mean  $\\theta_v$ and Mean wind",' K',True,None,2e3],
                                [-5,5,Wcm,'both','%.1f',None,"Mean W and thermal base",' m/s',True,'TBM',None] ]
                img_maps_list[1].append( plot_maps(fig_maps_SL,axs_maps_SL,text_title_SL,ZS,maps_VAR,maps_params, PRS,TBM,USL,VSL,legend=True if it == 0 else False ,map_name='SL') )
            
            # maps_VAR = [THV[0],BLH,1e3*RV[0]]
            # maps_params = [[292.5,322.5,'turbo','both','%d',None,"$\\theta_v$ and wind at 10m AGL",' K',True,None,'10m'],
            #                [0.,5000.,ZScm,'max','%d',[0,1000,2000,3000,4000],"Top of the mixed layer",' m',True,'CBM',None],
            #                [0,20,'YlGnBu','both','%.1f',None,"$r_v$ and wind at 10m AGL",' g/kg',True,None,'10m'] ]
            
            
            # maps_VAR = [RV[0],TCW,WBL]
            # maps_params = [[0,20,'YlGnBu','both','%.1f',None,"$r_v$ and wind at 10m AGL",' g/kg',True,None,'10m'],
            #                [0.,5000.,ZScm,'max','%d',[0,1000,2000,3000,4000],"Top of the mixed layer",' m',True,'CBM',None],
            #                [-5,5,'seismic','both','%.1f',None,"Mean W in the mixed layer",' m/s',True,None,None] ]
        
            elif maps_name=='gif_BLH':
                maps_VAR = [BLH]
                maps_params = [[0.,4000.,'turbo','max','%d',[0,1000,2000,3000,4000],"Top of the mixed layer",' m',False,None,None]]
                # maps_params = [[0.,5000.,'rainbow','max','%d',[0,1000,2000,3000,4000],"Top of the mixed layer",' m',True,None,None]]
                img_maps_list[i].append( plot_maps(fig_maps_list[i],axs_maps_list[i],text_title_list[i],ZS,maps_VAR,maps_params, PRS,TTM,UBL,VBL,legend=False,short_title=True,ZS_labels=False ,map_name='BLH') )
            elif maps_name=='gif_WBL':
                maps_VAR = [WBL]
                maps_params = [[-5,5,Wcm,'both','%.1f',None,"Mean W in mixed layer",' m/s',False,None,None] ]
                img_maps_list[i].append( plot_maps(fig_maps_list[i],axs_maps_list[i],text_title_list[i],ZS,maps_VAR,maps_params, PRS,TTM,UBL,VBL,legend=False ,short_title=True,ZS_labels=False,map_name='WBL') )
            elif maps_name=='gif_THVA2m':
                AGL = 2.
                AL = Zm[Zm<np.max(ZS)+1.1*np.max(dz)]
                maps_VAR = [AGL_lin_anomaly( AGL_lin_interp(THV,AGL,Zm,ZS) ,AGL,AL,np.nanmean(AL_lin_interp(THV,AL,Zm,ZS),axis=(1,2)) ,ZS )]
                maps_params = [[-4,4,'RdBu_r','both','%.1f',None,"Buoyancy $\\theta _v ^{\prime}$ at 2m",' K',False,None,None]]
                img_maps_list[i].append( plot_maps(fig_maps_list[i],axs_maps_list[i],text_title_list[i],ZS,maps_VAR,maps_params, PRS,TTM,UBL,VBL,legend=False,short_title=True ,ZS_labels=False,map_name='THVA2m') )
        
        print('Make and save maps :' ,str(round(time.time()-time0,1))+' s',end=' ') ; time0 = time.time()
        
    if make_hist2D:
        _ = plot_hist2D(Z,W_,TH_,TKE_,regions[1],regions[2])
        # name = plot_hist2D(Z,W_,TH_,TKE_,regions[1],regions[2])
        # img_hist2D.append(name)
        

#%%
if make_hori_mean:
    ds_dict = { "ZS": (("y", "x"), ZS,{"units":"m"}) , "dz":(("z",),dZ,{"units":"m"}) }
    for ir,name_reg in enumerate(region_names):
        if region_types[ir] == '2D':
            ds_dict[name_reg] = (("y", "x"), regions[ir],{"units":"bool"})
        for iv,name_var in enumerate(VAR3D_2mean_names):
            name = name_var if name_reg=='a' else name_var+'_'+name_reg
            ds_dict[name] = (("time","z"),data3D_mean[ir,iv,:],{"units":VAR3D_2mean_units[iv]})
        for iv,name_var in enumerate(VAR2D_2mean_names):
            name = name_var if name_reg=='a' else name_var+'_'+name_reg
            ds_dict[name] = (("time",),data2D_mean[ir,iv],{"units":VAR2D_2mean_units[iv]})
    
    if orog:
        ds_dict["MO_flux_RHO"] = (("time", "zm"), data_flux_MO[0],{"units":"kg/s"})
        ds_dict["MO_flux_RV"] = (("time", "zm"), data_flux_MO[1],{"units":"kg/s"})
        ds_dict["MO_flux_RT"] = (("time", "zm"), data_flux_MO[2],{"units":"kg/s"})
        ds_dict["MO_flux_TH"] = (("time", "zm"), data_flux_MO[3],{"units":"K/s"})
            
    ds = xr.Dataset( ds_dict,coords={"time": times,"zm":z,"z": Z,"y": y,"x": x},)
    if not os.path.exists(savePathMean): os.makedirs(savePathMean) ; print('savePath directory created !')
    save_name = savePathMean+simu+'_'+seg+'_mean.nc'
    if os.path.exists(save_name): os.remove(save_name)
    ds.to_netcdf(save_name, encoding={"time": {'units': 'days since 1900-01-01'}})

#%%
plt.close("all")
if make_maps:
    from moviepy.editor import ImageSequenceClip
    
    # duration = 2
    # m0 = 60*12 + 0
    # m0 = 60*9 + 30
    # img = [savePathFigures+'maps_video/maps_video_{:02d}h{:02d}'.format((m0+m)//60,(m0+m)%60)+simu+'.png' for m in range(len(lFiles))]
    # img = [savePath+'maps_video/maps_video_{:03d}h'.format(h)+simu+'.png' for h in range(1,121)]
    
    # clips = [ImageClip(m).set_duration(2) for m in img]
    
    for i,img_maps in enumerate(img_maps_list):
        concat_clip = ImageSequenceClip(img_maps,  fps=5)
        if img_maps_names[i][:3]=='gif':
            concat_clip.resize(0.5).write_gif(savePathFigures+img_maps_names[i]+'_'+simu+'.gif', program='ffmpeg')
        concat_clip.write_videofile(savePathFigures+img_maps_names[i]+'_'+simu+'.mp4')

#%%
if make_sparses:
    for var in ['rcloud','rprecip','w','thva']:#,'rtracer'
        var_sparses = [ sparse.load_npz(savePathSkyView+var+'_sparse_'+simu+'_'+str(it)+'.npz' ) for it in range(nt) ]
        var_sparse = sparse.concatenate( var_sparses ,axis=0)
        sparse.save_npz(savePathSkyView+var+'_sparse_'+simu ,var_sparse)  
        print(var,end=' ')
        emptyness(var_sparse)
        
        for it in range(nt):
            os.remove(savePathSkyView+var+'_sparse_'+simu+'_'+str(it)+'.npz' )
    
#%%
# import matplotlib.pyplot as plt
# fig,axs = plt.subplots(ncols=5)
# for it,tt in enumerate(times):
#     axs[0].plot(data3D_mean[0,it],Z,label=tt)
#     axs[1].plot(data3D_mean[1,it],Z,label=tt)
#     axs[2].plot(data3D_mean[2,it],Z,label=tt)
#     axs[3].plot(data3D_mean[3,it],Z,label=tt)
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
    
# plot_profiles(z,data3D_mean_MO)
# plot_profiles(z,data3D_mean_PL)
# plot_profiles(z,data_flux_MO)

#%%
# for it,tt in enumerate(times):
#     for i in range(4):
#         print( np.sum(dz*data_flux_MO[i,it]) / np.sum(dz*np.abs(data_flux_MO[i,it])) )
        
#%%
# fig,ax = plt.subplots()
# ax2 = ax.twiny()
# for it,tt in enumerate(times):
#     ax.plot(data3D_mean_MO[2,it]-data3D_mean_PL[2,it] , z)
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