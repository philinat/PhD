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
Wcm = matplotlib.colors.LinearSegmentedColormap.from_list('seismic2',[(0,(0.2,0.,0.3)),(0.25,(0,0,1)),(0.28,(0,0.2,1)),(0.45,(0,0.9,1)),(0.5,(1,1,1)),(0.55,(1,0.9,0)),(0.73,(1,0.2,0)),(0.75,(1,0,0)),(1,(0.3,0.,0.2))])
 
# simu = 'A0W0V'
# simu = 'ATRI2'
# simu = 'AMOPL'
# simu = 'ACON2'
# simu='HUM25'

# simu = 'DFLAT'
# simu = 'DF500'
# simu = 'DMO10'
# simu = 'DTRI2'
simu = 'DMOPL'
seg = 'M100m'
# seg = '1D164'
# seg = '1D292'
zmax = 5 # km

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
    # precip = np.array(f.ACPRRSTEP[:])
    
if model=='LES':
    if userPath == '/home/philippotn':
        dataPath = userPath+'/Documents/NO_SAVE/'
        # dataPath = userPath+'/Documents/SIMU_LES/'
        savePath = userPath+'/Images/LES_'+simu+'/Plot_horizontal_mean/'
    else:
        dataPath = userPath+'/LES_'+simu+'/NO_SAVE/'
        savePath = userPath+'/LES_'+simu+'/Plot_horizontal_mean/'
        
    f = xr.open_dataset(dataPath+simu+'_'+seg+'_mean.nc')
    ZS = f['ZS']
    if np.all(ZS==ZS[0,0]): ZS=None
    z = f['z']/1000
    theta = f['TH']
    rvap = f['RV']
    cloud_frac = f['CF']
    pressure = f['P']
    temp = theta * (pressure/100000)**(2/7)
    rtracer = f['TR']
    # precip = f['PRS']
    tke = f['TKE']
    thetaflux = f['WTH']
    
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

theta_ = np.copy(theta)
THmin = np.min(theta) ; THmax = np.max(theta[:,z<zmax])
if simu[0]=='D': THmin = 300.
theta_contour = np.arange(int(THmin),int(THmax),1)
clfr_contour = np.array([0.,0.01,0.1,1.])

if cbtype == 'lin': 
    cbs_delta_theta=['RdBu_r',cbtype,'both',-2,2]
    cbs_delta_rvap=['BrBG',cbtype,'both',-2,2]
elif cbtype == 'asinh':
    cbs_delta_theta=['RdBu_r',cbtype,'both',-10,10,0.02]
    cbs_delta_rvap=['BrBG',cbtype,'both',-10,10,0.02]
    
def plot_basics(region):
    plot_hori_mean(var=theta,cbs=['turbo','lin','both',THmin,THmax],contour=theta_contour,var_contour=theta_,iso=False,
                saveName='theta_'+model+'_'+simu+region+'.png',varUnit='K',mathTitle = '$ \overline{ \\theta}^{'+region+'}$ and $\overline{\\theta}$ isolines')#,textTitle='Moyenne horizontale de la température potentielle')

    plot_hori_mean(var=delta_theta,cbs=cbs_delta_theta,contour=theta_contour,var_contour=theta_,contour_fmt = '%d',contour_label_step=5,
                    saveName='trend_theta_'+model+'_'+simu+region+'.png',varUnit='K/h',mathTitle='$ \overline{ \\frac{\\partial \\theta}{\\partial t}}^{'+region+'}$ and $\overline{\\theta}$ isolines')#,textTitle='Moyenne horizontale de la tendance en température potentielle' )
    
    # plot_hori_mean(var=delta_rvap,cbs=cbs_delta_rvap,contour=clfr_contour,var_contour=cloud_frac,contour_fmt = '%.0f%%', ### cloud fraction isolines
    plot_hori_mean(var=delta_rvap,cbs=cbs_delta_rvap,contour=theta_contour,var_contour=theta_,contour_fmt = '%d',contour_label_step=5,
                    saveName='tend_rv_'+model+'_'+simu+region+'.png',varUnit='g/kg/h',mathTitle='$ \overline{ \\frac{\\partial r_v}{\\partial t}}^{'+region+'}$ and $\overline{\\theta}$ isolines')#,textTitle="Moyenne horizontale de la tendance en vapeur d'eau" )
    
    plot_hori_mean(var=rtracer,cbs=['YlGn','log','both',1e-3,1e2],contour=theta_contour,var_contour=theta_,contour_fmt = '%d',contour_label_step=5,
                    saveName='mean_tr_'+model+'_'+simu+region+'.png',varUnit='g/kg',mathTitle='$ \overline{ tracer }^{'+region+'}$ and $\overline{\\theta}$ isolines')#,textTitle="Moyenne horizontale des traceurs" )
    
    plot_hori_mean(var=tke,cbs=['magma_r','log','both',1e-3,1e1],contour=theta_contour,var_contour=theta_,contour_fmt = '%d',contour_label_step=5,
                    saveName='mean_tke_'+model+'_'+simu+region+'.png',varUnit='$m^2 / s^2$',mathTitle='$ \overline{ tke }^{'+region+'}$ and $\overline{\\theta}$ isolines')
    
    plot_hori_mean(var=thetaflux,cbs=['PiYG','lin','both',-0.3,0.3],contour=theta_contour,var_contour=theta_,contour_fmt = '%d',contour_label_step=5,
                    saveName='mean_wth_'+model+'_'+simu+region+'.png',varUnit='K·m/s',mathTitle='$ \overline{ w·\\theta ^{\prime} }^{'+region+'}$ and $\overline{\\theta}$ isolines')
    plt.close("all")

plot_basics('')

#%%
def plot_regions(region):
    plot_hori_mean(var=alpha,cbs=['RdPu','lin','max',0,np.max(alpha)],contour=theta_contour,var_contour=theta_,contour_fmt = '%d',contour_label_step=5,
                    saveName='alpha_'+model+'_'+simu+region+'.png',varUnit='',mathTitle='$ \\alpha ^{'+region+'}$ and $\overline{\\theta}$ isolines')
    plot_hori_mean(var=w,cbs=[Wcm,'lin','both',-3,3],contour=theta_contour,var_contour=theta_,contour_fmt = '%d',contour_label_step=5,
                    saveName='w_'+model+'_'+simu+region+'.png',varUnit='m/s',mathTitle='$ \overline{ w }^{'+region+'}$ and $\overline{\\theta}$ isolines')
    plt.close("all")
    
if model=='LES':
    regions = ["e","t","s",'MO','PL'] if ZS is not None else ["e","t","s"]
    for region in regions:
        theta = f['TH_'+region] ; delta_theta = trend(theta) *3600
        rvap = f['RV_'+region] ; delta_rvap = trend(rvap) *3600*1000
        cloud_frac = f['CF_'+region]
        pressure = f['P_'+region]
        temp = theta * (pressure/100000)**(2/7)
        rtracer = f['TR_'+region]
        # precip = f['PRS_'+region]
        tke = f['TKE_'+region]
        thetaflux = f['WTH_'+region]
        plot_basics('_'+region)
        
        alpha = f['ALPHA_'+region]
        w = f['W_'+region]
        plot_regions('_'+region)
#%%
if model=='LES' and ZS is not None:
    cbs_massflux=['RdBu_r',cbtype,'both',-5000,5000]
    cbs_dpressure=['RdBu_r',cbtype,'both',-2,2]
    
    massflux_contour = np.array([0.,0.01,0.1,1.])
    massflux = f['MO_flux_RHO']
    delta_MO_PL_thetav = f['THV_MO'] - f['THV_PL']
    delta_MO_PL_pressure = (f['P_MO'] - f['P_PL']) / 100 # hPa
    
    plot_hori_mean(var=delta_MO_PL_thetav ,cbs=cbs_delta_theta, 
                saveName='dthetav_MO_PL_'+model+'_'+simu+'.png',varUnit='K',mathTitle='$ \overline{  \\theta}_v^{MO} - \overline{  \\theta}_v^{PL}$')#,textTitle='Moyenne horizontale de la tendance en température potentielle' )
    
    plot_hori_mean(var=delta_MO_PL_pressure ,cbs=cbs_dpressure, 
                saveName='dpressure_MO_PL_'+model+'_'+simu+'.png',varUnit='K',mathTitle='$ \overline{ p}^{MO} - \overline{p}^{PL}$')#,textTitle='Moyenne horizontale de la tendance en température potentielle' )
    
    plot_hori_mean(var=massflux ,cbs=cbs_massflux, 
                saveName='massflux_MO_PL_'+model+'_'+simu+'.png',varUnit='kg/s',mathTitle='Massflux PL--> MO')#,textTitle='Moyenne horizontale de la tendance en température potentielle' )

plt.close("all")