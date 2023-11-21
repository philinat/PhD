#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:38:41 2023

@author: philippotn
"""

# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xarray as xr
from numba import njit
import os
from matplotlib.colors import LightSource
ls = LightSource(azdeg=325,altdeg=45)
import warnings
warnings.filterwarnings('ignore')

simu = 'AMOPL'

# userPath = '/cnrm/tropics/user/philippotn'
userPath = '/home/philippotn'

if userPath == '/home/philippotn':
    dataPath = userPath+'/Documents/SIMU_LES/'
    savePath = userPath+'/Images/LES_'+simu+'/figures_maps/'
else:
    dataPath = userPath+'/LES_'+simu+'/SIMU_LES/'
    savePath = userPath+'/LES_'+simu+'/figures_maps/'
    
if simu == 'AMOPL':
    # lFiles = [dataPath + 'AMOPL.1.R200m.OUT.{:03d}.nc'.format(i) for i in range(100,601,100)]
    lFiles = [dataPath + 'AMOPL.1.200m1.OUT.{:03d}.nc'.format(i) for i in [60,240]]+[dataPath + 'AMOPL.1.200m2.OUT.{:03d}.nc'.format(i) for i in [180,360,540,720]]
    # lFiles = [dataPath + 'AMOPL.1.200m1.OUT.{:03d}.nc'.format(i) for i in range(1,241,1)] + [dataPath + 'AMOPL.1.200m2.OUT.{:03d}.nc'.format(i) for i in range(1,722,1)]



print('lecture du fichier')
f0 = xr.open_dataset(lFiles[-1])
print('fin de la lecture')

nx1 = 1 ; nx2 = len(f0.ni)-1
ny1 = 1 ; ny2 = len(f0.nj)-1
nz1 = 1 ; nz2 = len(f0.level)

x = np.array(f0.ni)[nx1:nx2]
y = np.array(f0.nj)[ny1:ny2]
z = np.array(f0.level)[nz1:nz2]

dx=x[1]-x[0]
nz,nx,ny = len(z),len(x),len(y)

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

ZS = xr.open_dataset(dataPath+simu+'_init_R'+str(round(dx))+'m_pgd.nc')['ZS'][ny1:ny2,nx1:nx2].data


    
def find_zind(alti):
    for iz in range(len(z)):
        if z[iz]>alti:
            return iz
    return len(z)-1

@njit()
def top_cloud(z,r):
    ny,nx = np.shape(r[0])
#    ztop = -np.ones((ny,nx))
    ztop = np.full((ny,nx),np.nan)
    for j in range(ny):
        for i in range(nx):
            for h in range(len(z)-1,-1,-1):
                if r[h,j,i]>1e-6:
                    ztop[j,i] = z[h]
                    break
                if h==0:
                    ztop[j,i] = -1e10
    return ztop

@njit()
def boundary_layer_height(z,ZS,rc,thv,rho):
    ny,nx = np.shape(rc[0])
    BLH = np.full((ny,nx),np.nan)
    cloud_base_mask = np.zeros((ny,nx),dtype=np.int32)
    ZTOP = z[-1]
    for j in range(ny):
        for i in range(nx):
            mean_thv = thv[0,j,i]
            mass = z[0]*rho[0,j,i]
            for h in range(1,len(z)-1):
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
                    mass += layer_mass
    return BLH,cloud_base_mask

from numpy.lib.stride_tricks import as_strided
def regrid_mean(a,n):
    return as_strided(a, shape= tuple(map(lambda x: int(x / n), a.shape)) + (n, n), strides=tuple(map(lambda x: x*n, a.strides)) + a.strides).mean(axis=(2,3))

@njit()
def object_top(objects):
    nz,nx,ny = objects.shape
    nobj = np.max(objects)+1
    top = (nz-1) * np.ones(nobj,dtype=np.int32)
    for h in range(nz-2,-1,-1):
        for i in range(nx):
            for j in range(ny):
                obj = objects[h,i,j]
                if obj>0 and top[obj]==nz-1: # Compute mean position only for clouds (not for environment)
                    top[obj] = h
    return top
#%%
# def plot_video(lFiles):
ft=20
fig=plt.figure(figsize=(16.2,10.8))
ax=fig.add_axes([0.05,0., 2/3, 1.])
    
for it,fname in enumerate(lFiles):
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim([x[0]/1000,x[-1]/1000])
    ax.set_ylim([y[0]/1000,y[-1]/1000])
    ax.set_xticks([])
    plt.yticks(fontsize=ft*2/3)
    ax.set_ylabel('km',fontsize=ft,rotation=0)
    print(fname)

    ## Definition des variables à plotter
    f = xr.open_dataset(fname)
    def npf(v):
        return f[v][0,nz1:nz2,ny1:ny2,nx1:nx2].data
    
    print("Start",end='')
    # cloud = f.clouds_w[0].data
    TH = npf('THT')
    RV = npf('RVT')
    RC = npf('RCT') + npf('RIT')
    RP = npf('RRT')# + npf('RST') + npf('RGT')
    P = npf('PABST')
    
    THV = TH * (1+ 1.61*RV)/(1+RV+RC+RP)
    Rd = 287.04     # J/kg/K 
    RHO = P/Rd/ (THV * (P/100000)**(2/7))
    BLH,cloud_base_mask = boundary_layer_height(z,ZS,RC,THV,RHO)
    
    precip = f.ACPRRSTEP[0,ny1:ny2,nx1:nx2].data
    flat_precip = get_flat_precip(x,y,flat_X,flat_Y,precip)
    select_precip = flat_precip > 10**-4 # 1 mm of rain in 1 min
    
    # SURF = 1
    # THV_SURF = TH[SURF] * (1+ 1.61*RV[SURF])/(1+RV[SURF]+RP[SURF])
    # anom_thetav_SURF = thetav_SURF - np.mean(thetav_SURF,axis=(0,1))
    
    
    print(" ---- Surface ready",end='')
    BLH_cf = ax.imshow(BLH,cmap='rainbow',visible=False,vmin=0.,vmax=4000.,origin='lower',extent=[x[0]/1000,x[-1]/1000,y[0]/1000,y[-1]/1000])
    # BLH_cf = ax.pcolormesh(X/1000,Y/1000,BLH,vmin=0.,vmax=4000.,cmap='rainbow',shading='gouraud')
    
    

    cmap = matplotlib.cm.get_cmap('rainbow')
    # ccmap = im.get_cmap() #it is a function
    im_rgb = cmap(BLH/4000)[:,:,:3]
    rgb = ls.shade_rgb(im_rgb,ZS,blend_mode='soft',vert_exag=0.1 )
    ax.imshow( rgb ,origin='lower',extent=[x[0]/1000,x[-1]/1000,y[0]/1000,y[-1]/1000] )
    # T_cf = ax.pcolormesh(X/1000,Y/1000,anom_thetav_SURF,norm=matplotlib.colors.AsinhNorm(linear_width=0.15,vmin=-6, vmax=6),cmap='RdBu_r',shading='gouraud')

    ax.scatter(flat_X[select_precip]/1000, flat_Y[select_precip]/1000, s= flat_precip[select_precip]*5e2, c='black',zorder=100)
    
    ax.contour(X/1000,Y/1000,cloud_base_mask, levels=[0.5],linewidths=2,alpha=1,colors=['k'])#,colors=[CTcm( (z[cloud_top[obj]]/1000 -ct_lev[0])/ct_lev[-1] )])
    
    if fname == lFiles[0]:
        cax=fig.add_axes([0.75,0.02, 0.03, 0.7])
        
        BLH_cb = plt.colorbar(BLH_cf,cax=cax,orientation = "vertical",extend='both',format='%.0f')
        # BLH_cb.set_ticks([-5,-2,-1,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1.,2.,5.])
        BLH_cb.ax.tick_params(labelsize=ft*2/3)
        BLH_cb.ax.set_title("$Top of BL$",fontsize=ft)
        BLH_cb.set_label(' m',fontsize=ft,rotation=0)
        
        # T_cb = plt.colorbar(T_cf,cax=cax,orientation = "vertical",extend='both',format='%.1f')
        # T_cb.set_ticks([-5,-2,-1,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1.,2.,5.])
        # T_cb.ax.tick_params(labelsize=ft*2/3)
        # T_cb.ax.set_title("$\\theta _v ^{\prime}$",fontsize=ft)
        # T_cb.set_label(' K',fontsize=ft,rotation=0)
        
        # cax=fig.add_axes([0.90,0.02, 0.03, 0.7])
        # ct_cb = plt.colorbar(ax.pcolormesh(X/1e6,Y/1e6,cloud_CL,cmap=CTcm,vmin=ct_lev[0],vmax=ct_lev[-1],alpha=1.) ,cax=cax,orientation = "vertical",extend='max')# ,ticks=ct_lev[::8],pad=cb_pad,shrink=cb_shrink,orientation = "vertical",extend='max')
        # ct_cb.set_ticks(ct_lev[::16])
        # ct_cb.ax.tick_params(labelsize=ft*2/3)
        # ct_cb.ax.set_title("Cloud top",fontsize=ft)
        # ct_cb.set_label('  km',fontsize=ft,rotation=0)
        
        cax=fig.add_axes([0.75,0.84, 0.2, 0.03])
        cax.set_xlim([0.5,4.5])
        cax.set_ylim([0.,2.])
        cax.set_yticks([])
        cax.set_xticks([1,2,3,4])
        cax.set_xticklabels(['0.1','1','10','100'],size=ft*2/3)
        # cax.xticks.label.set_fontsize(2/3*ft) 
        cax.scatter([1,2,3,4],[1,1,1,1],s=0.5*np.array([0.1,1,10,100]), c='black')
        cax.set_xlabel('mm / min',fontsize=ft)
        cax.set_title('Rain at the surface',fontsize=ft)
        
        text_time = fig.text(0.85,0.96,'{} h {:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute)),fontsize=1.5*ft,ha='center')
        # text_LCL = fig.text(0.85,0.93,'LCL = {:.1f} m'.format(zCL),fontsize=ft,ha='center')
    
    # text_time.set_text('{} h'.format(int(hours[it])))
    text_time.set_text('{} h {:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute)))
    # text_LCL.set_text('LCL = {:.1f} m'.format(zCL))
    print(" ---- Plot done",end='')
    if not os.path.exists(savePath+'maps_video/'): os.makedirs(savePath+'maps_video/') ; print('Directory created !')
    plt.savefig(savePath+'maps_video/maps_video_{:02d}h{:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute))+simu+'.png')
    # plt.savefig(savePath+'maps_video/maps_video_{:03d}h'.format(int(hours[it]))+simu+'.png')
    print(' ---- Saved {:02d}h{:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute)))
    
        
    
# plot_video(lFiles[:]) 


#%%
# from moviepy.editor import ImageSequenceClip

# # duration = 2
# # m0 = 60*12 + 0
# m0 = 60*9 + 30
# img = [savePath+'maps_video/maps_video_{:02d}h{:02d}'.format((m0+m)//60,(m0+m)%60)+simu+'.png' for m in range(1,361+240)]
# # img = [savePath+'maps_video/maps_video_{:03d}h'.format(h)+simu+'.png' for h in range(1,121)]

# # clips = [ImageClip(m).set_duration(2) for m in img]
# concat_clip = ImageSequenceClip(img,  fps=10)
# concat_clip.write_videofile(savePath+'video_cold_pool_'+simu+'.mp4')
#%%
        # precip = f.RRT.data[0] + f.RST.data[0] + f.RGT.data[0]
        # flat_precip = get_flat_precip(x,y,flat_X,flat_Y,cloud_water_path(precip))
        # select_precip = flat_precip > 10**-6
        # ax.scatter(flat_X[select_precip]/1000, flat_Y[select_precip]/1000, s= flat_precip[select_precip]/10, c='black')
        
        
        # # wp_cf = ax.contourf(X/1000,Y/1000,cloud_water_path(cloud), norm=matplotlib.colors.LogNorm(),levels=wp_lev,cmap='Blues_r',extend='both')  
        # wp_cf = ax.pcolormesh(X/1000,Y/1000,cloud_water_path(cloud), norm=matplotlib.colors.LogNorm(),cmap='Greys_r')
        # if colorbars and fname == lFiles[0]:
        #     wp_cb = plt.colorbar(wp_cf,ax=ax ,ticks=[10**-3,10**-2,10**-1,1,10],pad=cb_pad,shrink=cb_shrink,orientation = "vertical",extend='both')
        #     wp_cb.ax.tick_params(labelsize=ft*2/3)
        #     wp_cb.ax.set_title("kg/m²",fontsize=ft)
        
        # # CT = top_cloud(z,cloud)/1000
        # ct_cf = ax.contourf(X/1000,Y/1000,CT,levels=ct_lev,cmap=CTcm,linewidths=0.,alpha=1.,antialiased=True) 
        # # ct_cf = ax.contourf(X/1000,Y/1000,CT,levels=ct_lev[::16],cmap=CTcm,linewidths=0.,alpha=0.2,antialiased=True)  
        # # for ic in range(9):
        # #     ax.contour(X/1000,Y/1000,CT>=ct_lev[ic*16],levels=[0.5],colors=[CTcm(ic/8)],linewidths=0.9,alpha=1,antialiased=True)  
        # if colorbars and fname == lFiles[0]:
        #     cax=fig.add_axes([0.85,0.05, 0.03, 0.7])
        #     ct_cb = plt.colorbar(ct_cf,ax=ax,pad=cb_pad,shrink=cb_shrink,orientation = "vertical",extend='max')
        #     # ct_cb = plt.colorbar(ax.contour(X,Y,CT,levels=ct_lev[::16],cmap=CTcm,linewidths=4.,alpha=1.) ,cax=ct_cb.ax)# ,ticks=ct_lev[::8],pad=cb_pad,shrink=cb_shrink,orientation = "vertical",extend='max')
        #     # ct_cb = plt.colorbar(ax.contour(X,Y,CT,levels=ct_lev[::16],cmap=CTcm,linewidths=4.,alpha=1.) ,ax=ax,pad=cb_pad,shrink=cb_shrink,orientation = "vertical",extend='max')# ,ticks=ct_lev[::8],pad=cb_pad,shrink=cb_shrink,orientation = "vertical",extend='max')
        #     ct_cb.set_ticks(ct_lev[::16])
        #     ct_cb.ax.tick_params(labelsize=ft*2/3)
        #     ct_cb.ax.set_title("Cloud top",fontsize=ft)
        #     ct_cb.set_label('km',fontsize=ft)
            
            
        # conv = horizontal_convergence(f.UT[0,SURF].data.T , f.VT[0,SURF].data.T ,dx)
        # C_cf = ax.contourf(X/1000,Y/1000, conv*100 , levels=C_lev,cmap='RdBu_r',extend='both')
        # if colorbars and fname == lFiles[0]:
        #     C_cb = plt.colorbar(C_cf,ax=ax ,ticks=C_lev[::10],pad=cb_pad,shrink=cb_shrink,orientation = "vertical")
        #     C_cb.ax.tick_params(labelsize=ft*2/3)
        #     C_cb.ax.set_title("m/s/100m",fontsize=ft)
            
        # # Cl_cf = ax.contourf(X/1000,Y/1000, conv*100 , levels=Cl_lev,cmap='summer',extend='max')s
        # Cl_cf = ax.contour(X/1000,Y/1000, conv*100 , levels=[0.5,1,2],colors=['green','yellow','violet'],extend='max',linewidths=[0.8,1,2])
        # if colorbars and fname == lFiles[0]:
        #     # Cl_cb = plt.colorbar(Cl_cf,ax=ax ,ticks=Cl_lev[::10],pad=cb_pad,shrink=cb_shrink,orientation = "vertical")
        #     Cl_cb = plt.colorbar(Cl_cf,ax=ax ,pad=cb_pad,shrink=cb_shrink,orientation = "vertical",format='%.1f')
        #     Cl_cb.ax.tick_params(labelsize=ft*2/3)
        #     Cl_cb.ax.set_title("m/s/100m",fontsize=ft)