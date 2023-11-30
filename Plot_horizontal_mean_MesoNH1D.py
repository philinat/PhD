#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:19:52 2023

@author: philippotn
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import os

simu = 'AMOPL'
model = 'MesoNH_1D'
# model = 'LES'
zmax = 16 # km
userPath = '/home/philippotn'

def trend(var):
    var_shift = np.roll(var,1,0)
    var_shift[0,:] = var[0,:]
    delta_var = (var-var_shift)/dt
    return delta_var

if model=='MesoNH_1D':
    dataPath = userPath+'/MNH-V5-6-1/MY_RUN/'+simu+'/002_mesonh/'
    savePath = userPath+'/MNH-V5-6-1/MY_RUN/'+simu+'/004_python/Plot_horizontal_mean/'
    
    f = xr.open_dataset(dataPath+simu+'.1.1D100.OUT.nc')
    time = np.array(f.time.dt.second + 60*f.time.dt.minute + f.time.dt.hour*3600)
    dt = time[1]-time[0]
    nz1 = 1 ; nz2 = len(f.level)-1
    z = f.level[nz1:nz2].data
    def npf(v,dtype=np.float32):
        return np.array(f[v][:,nz1:nz2],dtype=dtype)
    theta = npf('THT')
    rvap = npf('RVT')*1000
    cloud_frac = npf('CLDFR')
    pressure = npf('PABST')
    temp = theta * (pressure/100000)**(2/7)
    ZS = None
    # rtracer = npf('SVT001')
    
if model=='LES':
    if userPath == '/home/philippotn':
        dataPath = userPath+'/Documents/SIMU_LES/'
        savePath = userPath+'/Images/LES_'+simu+'/Plot_horizontal_mean/'
    else:
        dataPath = userPath+'/LES_'+simu+'/SIMU_LES/'
        savePath = userPath+'/LES_'+simu+'/Plot_horizontal_mean/'
        
    f = np.load(dataPath+simu+'_200_mean.npz')
    ZS = f['ZS']
    time = f['T']*3600
    dt = time[1]-time[0]
    z = f['Z']
    theta = f['TH']
    rvap = f['RV']
    cloud_frac = f['CF']
    # pressure = f['P']
    # temp = theta * (pressure/100000)**(2/7)
    rtracer = f['TR']
    
(T,Z) = np.meshgrid(time/3600,z/1000)
delta_theta = trend(theta)
delta_rvap = trend(rvap)
if not os.path.exists(savePath): os.makedirs(savePath) ; print('Directory created !')

#%%
def plot_hori_mean(var, saveName, cbs=['Greys','lin','both',-1.,1.], varUnit='', mathTitle='', textTitle='',iso=False,orog=True,ft=20,fmt='%g',#'%.2f',
                   contour=None,var_contour=None,contour_fmt='%g',contour_label_step=1): 
    print(saveName)
    plt.figure(figsize=(17,10),constrained_layout=True)
    
    if cbs[1]=='lin':
        norm=matplotlib.colors.Normalize(vmin=cbs[3], vmax=cbs[4])
    elif cbs[1]=='log':
        norm=matplotlib.colors.LogNorm(vmin=cbs[3], vmax=cbs[4])
    elif cbs[1]=='asinh':
        norm=matplotlib.colors.AsinhNorm(linear_width=cbs[5],vmin=cbs[3], vmax=cbs[4])
        
    # cf = plt.contourf(T,Z,var.T,levels=100,norm=norm,cmap=cbs[0],extend=cbs[2])
    cf = plt.pcolormesh(T,Z,var.T,norm=norm,cmap=cbs[0])#,shading='gouraud')
    cb = plt.colorbar(cf,aspect=30,format=fmt,extend=cbs[2],pad=0.02)
    cb.set_label(varUnit,fontsize=ft)

    if contour is not None :
        print("contour")
        CS = plt.contour(T,Z,var_contour.T,levels=contour,alpha=0.5,colors='Grey')
        labels = CS.levels[::contour_label_step]
        if contour_fmt == '%.0f%%': labels *= 100
        plt.clabel(CS, CS.levels[::contour_label_step], inline=True, inline_spacing=0, fmt=contour_fmt, fontsize=10)
        
    if iso :
        print("isotherme")
        plt.contour(T,Z,temp.T,levels=[273.15],colors='b')
        plt.text((time[0]+(time[-1]-time[0])/10)/3600,z[np.argmin(np.array(np.abs(temp[0]-273.15)))]/1000,'iso 0°C',fontsize=ft,backgroundcolor='w',color='b', ha="center", va="center")
    if orog and ZS is not None:
        plt.axhline(np.mean(ZS)/1000,linestyle='--',color='k',lw=2,label='mean(Zs)')
        plt.axhline(np.max(ZS)/1000,linestyle=':',color='k',lw=2,label='max(Zs)')
        plt.legend(fontsize=ft)
        
    plt.title(mathTitle,fontsize=1.5*ft)
    plt.suptitle(textTitle+' '+model+' '+simu+' ',fontsize=ft)
    plt.ylim([0,zmax])
    plt.xlim([np.min(time/3600),np.max(time/3600)])
    plt.xticks(fontsize=ft)
    plt.yticks(fontsize=ft)
    plt.ylabel('Altitude (km)',fontsize=ft)
    plt.xlabel('Day time (hours)',fontsize=ft)
    plt.savefig(savePath+saveName)
    plt.show()


theta_contour = np.arange(270,400,1)
clfr_contour = np.array([0.,0.01,0.1,1.])
plot_hori_mean(var=theta,cbs=['rainbow','lin','both',295,365],contour=theta_contour,var_contour=theta,iso=False,
                saveName='theta_'+model+'_'+simu+'.png',varUnit='K',mathTitle = '$ \overline{ \\theta}$',textTitle='Moyenne horizontale de la température potentielle')

plot_hori_mean(var=delta_theta*3600,cbs=['RdBu_r','asinh','both',-10,10,0.02],contour=theta_contour,var_contour=theta,contour_fmt = '%d',contour_label_step=5,
                saveName='trend_theta_'+model+'_'+simu+'.png',varUnit='K/h',mathTitle='$ \overline{ \\frac{\\partial \\theta}{\\partial t}}$ and $ \overline{\\theta}$ isolines')#,textTitle='Moyenne horizontale de la tendance en température potentielle' )

plot_hori_mean(var=delta_rvap*3600,cbs=['BrBG','asinh','both',-10,10,0.02],contour=clfr_contour,var_contour=cloud_frac,contour_fmt = '%.0f%%',
                saveName='tend_rv_'+model+'_'+simu+'.png',varUnit='g/kg/h',mathTitle='$ \overline{ \\frac{\\partial r_v}{\\partial t}}$ and cloud fraction isolines')#,textTitle="Moyenne horizontale de la tendance en vapeur d'eau" )

# plot_hori_mean(var=rtracer,cbs=['YlGn','log','both',1e-12,1e2],contour=clfr_contour,var_contour=cloud_frac,contour_fmt = '%.0f%%',
#                 saveName='mean_tr_'+model+'_'+simu+'.png',varUnit='g/kg',mathTitle='$ \overline{ Tracer }$ and cloud fraction isolines')#,textTitle="Moyenne horizontale de la tendance en vapeur d'eau" )

# plt.close("all")