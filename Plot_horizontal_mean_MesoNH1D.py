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
import locale
locale.setlocale(locale.LC_ALL,'en_US.utf8')

# simu = 'A0W0V'
simu = 'ATRI2'
# simu = 'ACON2'
seg = 'S200m'
# seg = '1D164'
zmax = 16 # km

# simu = 'CFLAT'
# simu = 'CTRI2'
# seg = 'S100m'
# zmax = 4 # km

# model = 'MesoNH_1D'
# model = 'LMDZ_1D'
model = 'LES'

# cbtype = 'asinh'
cbtype = 'lin'

userPath = '/home/philippotn'
# userPath = '/cnrm/tropics/user/philippotn'

if model=='LMDZ_1D':
    if simu[0]=='A': dataPath = userPath+'/Documents/LMDZ/LMDZseq/1D/EXEC/6AL79/AMMA/REF/'
    if simu[0]=='C': dataPath = userPath+'/Documents/LMDZ/LMDZseq/1D/EXEC/6AL79/ARMCU/REF/'
    savePath = userPath+'/Images/LMDZ_1D/'+simu+'/'
    
    # f = xr.open_dataset(dataPath+'histhf_'+simu[1:]+'.nc')
    f = xr.open_dataset(dataPath+'histhf_REF.nc')
    nt1 = 0 ; nt2 = -12
    # time = np.array(f.time_counter.dt.second + 60*f.time_counter.dt.minute + f.time_counter.dt.hour*3600)[nt1:nt2]
    z = np.array(f.Alt)*1000
    theta = np.array(f.theta)[nt1:nt2,:,0,0]
    rvap = np.array(f.ovap)[nt1:nt2,:,0,0]
    temp = f.temp[nt1:nt2,:,0,0]
    cloud_frac = f.rneb[nt1:nt2,:,0,0]
    ZS = None
    
if model=='MesoNH_1D':
    dataPath = userPath+'/MNH-V5-6-1/MY_RUN/'+simu+'/002_mesonh/OUT/'
    savePath = userPath+'/MNH-V5-6-1/MY_RUN/'+simu+'/003_python/'
    
    f = xr.open_dataset(dataPath+simu+'.1.'+seg+'.OUT.nc')
    nz1 = 1 ; nz2 = len(f.level)-1
    z = f.level[nz1:nz2].data/1000
    def npf(v,dtype=np.float32):
        return np.array(f[v][:,nz1:nz2],dtype=dtype)
    theta = npf('THT')
    rvap = npf('RVT')
    cloud_frac = npf('CLDFR')
    pressure = npf('PABST')
    temp = theta * (pressure/100000)**(2/7)
    ZS = None
    rtracer = npf('SVT001')
    precip = np.array(f.ACPRRSTEP[:])
    
if model=='LES':
    if userPath == '/home/philippotn':
        dataPath = userPath+'/Documents/SIMU_LES/'
        savePath = userPath+'/Images/LES_'+simu+'/Plot_horizontal_mean/'
    else:
        dataPath = userPath+'/LES_'+simu+'/NO_SAVE/'
        savePath = userPath+'/LES_'+simu+'/Plot_horizontal_mean/'
        
    f = xr.open_dataset(dataPath+simu+'_'+seg+'_mean.nc')
    ZS = f['ZS']
    if np.all(ZS==0.): ZS=None
    z = f['z']/1000
    theta = f['TH']
    rvap = f['RV']
    cloud_frac = f['CF']
    pressure = f['P']
    temp = theta * (pressure/100000)**(2/7)
    rtracer = f['TR']
    precip = f['PR']

time = f['time']
dt = float(time[1]-time[0])/1e9 # from ns to s
    
def trend(var):
    var_shift = np.roll(var,1,0)
    var_shift[0,:] = var[0,:]
    delta_var = (var-var_shift)/dt
    return delta_var
delta_theta = trend(theta) *3600
delta_rvap = trend(rvap) *3600*1000

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
        
    cf = plt.pcolormesh(time,z,var.T,norm=norm,cmap=cbs[0])#,shading='gouraud')
    cb = plt.colorbar(cf,aspect=30,format=fmt,extend=cbs[2],pad=0.02)
    cb.set_label(varUnit,fontsize=ft)

    if contour is not None :
        print("contour")
        CS = plt.contour(time,z,var_contour.T,levels=contour,alpha=0.5,colors='Grey')
        labels = CS.levels[::contour_label_step]
        if contour_fmt == '%.0f%%': labels *= 100
        plt.clabel(CS, CS.levels[::contour_label_step], inline=True, inline_spacing=0, fmt=contour_fmt, fontsize=10)
        
    if iso :
        print("isotherme")
        plt.contour(time,z,temp.T,levels=[273.15],colors='b')
        plt.text((time[0]+(time[-1]-time[0])/10)/3600,z[np.argmin(np.array(np.abs(temp[0]-273.15)))]/1000,'iso 0°C',fontsize=ft,backgroundcolor='w',color='b', ha="center", va="center")
    if orog and ZS is not None:
        plt.axhline(np.mean(ZS)/1000,linestyle='--',color='k',lw=2,label='mean(Zs)')
        plt.axhline(np.max(ZS)/1000,linestyle=':',color='k',lw=2,label='max(Zs)')
        plt.legend(fontsize=ft)
        
    plt.title(mathTitle,fontsize=1.5*ft)
    plt.suptitle(textTitle+' '+model+' '+simu+' ',fontsize=ft)
    plt.ylim([0,zmax])
    # plt.xlim([np.min(time/3600),np.max(time/3600)])
    
    # days = matplotlib.dates.DayLocator()
    hours = matplotlib.dates.HourLocator()
    ax = plt.gca()
    ax.xaxis.set_major_locator(hours)
    # ax.xaxis.set_minor_locator(days)
    ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(hours,formats=['%Y', '%b', '%b/%d', '%H', '%H:%M', '%S.%f']))
    # ax.xaxis.set_minor_formatter(matplotlib.dates.ConciseDateFormatter(days))
    plt.xticks(fontsize=ft)
    
    plt.yticks(fontsize=ft)
    plt.ylabel('Altitude (km)',fontsize=ft)
    plt.xlabel('Time UTC (hours)',fontsize=ft)
    plt.savefig(savePath+saveName)
    # plt.savefig(savePath+'dt10_'+saveName)
    plt.show()

THmin = np.min(theta) ; THmax = np.max(theta[:,z<zmax])
theta_contour = np.arange(int(THmin),int(THmax),1)
clfr_contour = np.array([0.,0.01,0.1,1.])
plot_hori_mean(var=theta,cbs=['rainbow','lin','both',THmin,THmax],contour=theta_contour,var_contour=theta,iso=False,
                saveName='theta_'+model+'_'+simu+'.png',varUnit='K',mathTitle = '$ \overline{ \\theta}$',textTitle='Moyenne horizontale de la température potentielle')

if cbtype == 'lin': 
    cbs_delta_theta=['RdBu_r',cbtype,'both',-2,2]
    cbs_delta_rvap=['BrBG',cbtype,'both',-4,4]
elif cbtype == 'asinh':
    cbs_delta_theta=['RdBu_r',cbtype,'both',-10,10,0.02]
    cbs_delta_rvap=['BrBG',cbtype,'both',-10,10,0.02]
plot_hori_mean(var=delta_theta,cbs=cbs_delta_theta,contour=theta_contour,var_contour=theta,contour_fmt = '%d',contour_label_step=5,
                saveName='trend_theta_'+model+'_'+simu+'.png',varUnit='K/h',mathTitle='$ \overline{ \\frac{\\partial \\theta}{\\partial t}}$ and $ \overline{\\theta}$ isolines')#,textTitle='Moyenne horizontale de la tendance en température potentielle' )

plot_hori_mean(var=delta_rvap,cbs=cbs_delta_rvap,contour=clfr_contour,var_contour=cloud_frac,contour_fmt = '%.0f%%',
                saveName='tend_rv_'+model+'_'+simu+'.png',varUnit='g/kg/h',mathTitle='$ \overline{ \\frac{\\partial r_v}{\\partial t}}$ and cloud fraction isolines')#,textTitle="Moyenne horizontale de la tendance en vapeur d'eau" )

plot_hori_mean(var=rtracer,cbs=['YlGn','log','both',1e-12,1e2],contour=clfr_contour,var_contour=cloud_frac,contour_fmt = '%.0f%%',
                saveName='mean_tr_'+model+'_'+simu+'.png',varUnit='g/kg',mathTitle='$ \overline{ Tracer }$ and cloud fraction isolines')#,textTitle="Moyenne horizontale des traceurs" )

plt.close("all")

#%%
plt.figure(11,figsize=(17,10),constrained_layout=True)
plt.plot(time,precip,color='b',label="Precipitations")
hours = matplotlib.dates.HourLocator()
ax = plt.gca()
ax.xaxis.set_major_locator(hours)
ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(hours,formats=['%Y', '%b', '%b/%d', '%H', '%H:%M', '%S.%f']))
ft = 20
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.ylabel('Precipitations (mm/h)',fontsize=ft)
plt.xlabel('Time UTC (hours)',fontsize=ft)
plt.savefig(savePath+'precip_'+model+'_'+simu+'.png')

#%%
if model=='LES' and ZS is not None:
    for region in ['MO','PL']:
        theta = f['TH_'+region]
        rvap = f['RV_'+region]
        cloud_frac = f['CF_'+region]
        pressure = f['P_'+region]
        temp = theta * (pressure/100000)**(2/7)
        rtracer = f['TR_'+region]
        precip = f['PR_'+region]
        
        delta_theta = trend(theta) *3600
        delta_rvap = trend(rvap) *3600*1000
        
        plot_hori_mean(var=delta_theta,cbs=cbs_delta_theta,contour=theta_contour,var_contour=theta,contour_fmt = '%d',contour_label_step=5,
                saveName='trend_theta_'+model+'_'+simu+'_'+region+'.png',varUnit='K/h',mathTitle='$ \overline{ \\frac{\\partial \\theta}{\\partial t}}$ and $ \overline{\\theta}$ isolines')#,textTitle='Moyenne horizontale de la tendance en température potentielle' )

        plot_hori_mean(var=delta_rvap,cbs=cbs_delta_rvap,contour=clfr_contour,var_contour=cloud_frac,contour_fmt = '%.0f%%',
                saveName='tend_rv_'+model+'_'+simu+'_'+region+'.png',varUnit='g/kg/h',mathTitle='$ \overline{ \\frac{\\partial r_v}{\\partial t}}$ and cloud fraction isolines')#,textTitle="Moyenne horizontale de la tendance en vapeur d'eau" )

        plot_hori_mean(var=rtracer,cbs=['YlGn','log','both',1e-12,1e2],contour=clfr_contour,var_contour=cloud_frac,contour_fmt = '%.0f%%',
                        saveName='mean_tr_'+model+'_'+simu+'_'+region+'.png',varUnit='g/kg',mathTitle='$ \overline{ Tracer }$ and cloud fraction isolines')#,textTitle="Moyenne horizontale des traceurs" )

plt.close("all")