#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:35:05 2022

@author: philippotn
"""
import time
import numpy as np
from numpy.lib.stride_tricks import as_strided
from matplotlib import pyplot as plt 
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import matplotlib.colors as mcolors

from matplotlib.lines import Line2D
from matplotlib.backend_tools import ToolToggleBase
plt.rcParams['toolbar'] = 'toolmanager'
#from matplotlib.backend_bases import MousseButton
import xarray as xr
from numba import njit,prange

floatype = np.float16

# - Data LES
simu = 'AMMA'
# simu = 'AMA0W'
# simu = 'BOMEX'
#simu = 'ABL0V'

relative_humidity = True
wind = True
put_in_RAM = True
orog = False

if simu == 'AMMA':
    path = '/cnrm/tropics/user/couvreux/POUR_NATHAN/AMMA/SIMU_LES/'
    lFiles = [path + 'AMMH3.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(300,302)]
    # lFiles = [path + 'AMMH3.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(294,322)]
    # lFiles = [path + 'AMMH3.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(315,322)]
    # else:
    #     lFiles = [path + 'AMMA2.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(1,7)] + [path + 'AMMH3.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(1,361)]
    
elif simu == 'AMA0W':
    path = '/cnrm/tropics/user/philippotn/LES_AMA0W/SIMU_LES/'
    lFiles = [path + 'AMA0W.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(1,7)] + [path + 'AMA0W.1.R6H1M.OUT.{:03d}.nc'.format(i) for i in range(1,361)]
    
elif simu == 'LBA':
    path = '/cnrm/tropics/user/couvreux/POUR_NATHAN/LBA/SIMU_LES/'
    lFiles = [path + 'LBA__.1.RFLES.OUT.{:03d}.nc'.format(i) for i in range(1,7)]
    tFile = path + 'new_LBA__.1.RFLES.000.nc'
    
elif simu == 'BOMEX':
    path = '/cnrm/tropics/user/couvreux/POUR_AUDE/SIMU_MNH/bigLES/'
    # lFiles = [path + 'L25{:02d}.1.KUAB2.OUT.{:03d}.nc'.format(i//6+1,i%6+1) for i in range(120)]
    hours = np.arange(100,121,1)
    lFiles = [path + 'L25{:02d}.1.KUAB2.OUT.{:03d}.nc'.format(i//6+1,i%6+1) for i in hours-1]

elif simu == 'ABL0V':
    path = '/cnrm/tropics/user/philippotn/LES_ABL0V/SIMU_LES/'
    lFiles = [path + 'ABL0V.1.SEG01.OUT.{:03d}.nc'.format(i) for i in range(200,301,20)]
    
savePath = '/cnrm/tropics/user/philippotn/LES_'+simu+'/figures_maps/'

f0 = xr.open_dataset(lFiles[0])

#%%

# nx1 = 1 ; nx2 = len(f0.ni)-1
# ny1 = 1 ; ny2 = len(f0.nj)-1
# nz1 = 1 ; nz2 = len(f0.level)-1

nx1 = 1 ; nx2 = len(f0.ni)*6//10
ny1 = 1 ; ny2 = len(f0.nj)*6//10
nz1 = 1 ; nz2 = len(f0.level)-24

x = np.array(f0.ni)[nx1:nx2]/1000
y = np.array(f0.nj)[ny1:ny2]/1000
z = np.array(f0.level)[nz1:nz2]/1000

dx=x[1]-x[0]
nz,nx,ny = len(z),len(x),len(y)

x_ = np.append(x,x[-1]+dx) -dx/2
y_ = np.append(x,x[-1]+dx) -dx/2
z_ = (np.array(f0.level)[nz1:nz2+1]+np.array(f0.level)[nz1-1:nz2])/2000
z__ = np.array(f0.level)[nz1-1:nz2+1]/1000

nt,nz,nx,ny = len(lFiles),len(z),len(x),len(y)

if orog:
    zs = xr.open_dataset(path+simu+'_init_pgd.nc')['ZS'][1:-1,1:-1].data
    
#%%
if relative_humidity:
    from compute_RH_lcl import RH_water_ice
if wind:
    new_z = np.linspace(dx,z[-1]-z[-1]%dx,int(z[-1]/dx))
    ios = np.zeros(len(new_z),dtype=int)
    c0s = np.zeros(len(new_z))
    c1s = np.zeros(len(new_z))
    for iz in range(len(new_z)):
        ios[iz] = np.argmin(np.abs(new_z[iz]-z))
        io=ios[iz]
        if new_z[iz]<z[io]:
            io-=1
        dz = z[io+1]-z[io]
        c0s[iz] = (z[io+1]-new_z[iz])/dz
        c1s[iz] = (new_z[iz]-z[io])/dz
        
    @njit()
    def interp_vertical(var,ios,c0s,c1s):
        nz,nx = np.shape(var)
        new_var = np.zeros((len(ios),nx),dtype=var.dtype)
        for iz in range(len(ios)):
            io=ios[iz]
            new_var[iz] = var[io]*c0s[iz] + var[io+1]*c1s[iz]
        return new_var
if orog:
    def interp_on_regular_grid(var,z,zs):
        return var
#%%
    
if not(put_in_RAM):
    times = []
    list_f = []
    for it,fname in enumerate(lFiles):
        f = xr.open_dataset(fname)
        list_f.append(f)
        times.append(' {:02d}h{:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute))) # year = str(f.time.data[0])[:10]+
        
else:
    def npf(v,dtype=floatype):
        return np.array(f[v][0,nz1:nz2,nx1:nx2,ny1:ny2],dtype=dtype)
    
    if wind:
        data5D = np.empty((8,nt,nz,nx,ny),dtype=floatype)
    else:
        data5D = np.empty((6,nt,nz,nx,ny),dtype=floatype)
    
    times = []
    for it,fname in enumerate(lFiles):
        time0 = time.time()
        f = xr.open_dataset(fname)
        times.append(' {:02d}h{:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute))) # year = str(f.time.data[0])[:10]+
        print(times[-1],end=' ')
        data5D[0,it] = npf('WT')
        data5D[2,it] = npf('RVT')
        data5D[3,it] = npf('RCT')+npf('RIT')
        data5D[4,it] = npf('RRT')+npf('RST')+npf('RGT')
        if wind:
            data5D[6,it] = npf('UT')
            data5D[7,it] = npf('VT')
        
        P = npf('PABST',dtype=np.float32)
        theta = npf('THT',dtype=np.float32)
        T = theta * (P/100000)**(2/7)
        ## Change potential temperature into buoyancy
        thv = theta * (1.+ 1.61*np.float32(data5D[2,it]))/(1.+np.float32(data5D[2,it]+data5D[3,it]+data5D[4,it]))
        # thv = theta * (1.+ 1.61*data5D[2,it])/(1.+data5D[2,it]+data5D[3,it]+data5D[4,it])
        data5D[1,it] = np.float16(thv - np.mean(thv,axis=(1,2),keepdims=True))
        # data5D[1,it] = thv - np.mean(thv,axis=(1,2),keepdims=True)
            
        if relative_humidity:
            rhl,rhi = RH_water_ice(data5D[2,it],T,P)
            data5D[2,it] = np.float16(np.maximum(rhl,rhi))
        else:
            data5D[2,it] *= 1000
        data5D[5,it] = np.float16(P - np.mean(P,axis=(1,2),keepdims=True))
        
        ##%% Change kg/kg into g/kg
        
        data5D[3,it] *= 1000
        data5D[4,it] *= 1000
        
        print(time.ctime()[-13:-5] ,str(round(time.time()-time0,1))+' s')
        
#%%
k = 2*np.pi*np.fft.rfftfreq(nx,dx)
m = 2*np.pi*np.fft.fftfreq(ny,dx)
kk,mm = np.meshgrid(k,m)
k2 = kk**2+mm**2
del(kk,mm)
def gaussian_filter(periodic_2Darray,lc):
    kc = 2*np.pi/lc
    return np.fft.irfft2(np.exp(-k2/kc**2/2)*np.fft.rfft2(periodic_2Darray) )

@njit()
def TDMAsolver(ac, bc, cc, dc):
    n = len(dc)
    for it in range(1, n):
        mc = ac[it]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
    bc[-1] = dc[-1]/bc[-1]
    for il in range(n-2, -1, -1):
        bc[il] = (dc[il]-cc[il]*bc[il+1])/bc[il]
    return bc

k = 2*np.pi*np.fft.rfftfreq(nx, d=dx)
nk = len(k)
kk , _ = np.meshgrid(k,np.zeros(nz))


b_diag = ( kk**2 + (( 2/(z__[2:]-z__[1:-1]) + 2/(z__[1:-1]-z__[:-2])  )/(z__[2:]-z__[:-2])).reshape((nz,1)) )*np.ones((nz,nk),dtype=complex)
a_side = ( -2/(z__[1:-1]-z__[:-2])/(z__[2:]-z__[:-2]) ).reshape((nz,1)) *np.ones((nz,nk),dtype=complex)
c_side = ( -2/(z__[2:]-z__[1:-1])/(z__[2:]-z__[:-2])  ).reshape((nz,1)) *np.ones((nz,nk),dtype=complex)
    
def gaussian_sp_fft_tridiag_filter(T,lc):
    kc2 = (2*np.pi/lc)**2
    # Tfilter = np.copy(T)
    Tf = np.fft.rfft(T)
    # Tf0 = Tf[0]# ; Tfn = Tf[-1]
    S = 2*kc2*Tf[:,:]
    # S[0,:] -= a_side[0]*Tf0
    # S[-1,:] -= a_side[-1]*Tfn
    Tf = TDMAsolver(a_side, 2*kc2 + b_diag , c_side , S)
    # T[1:-1,:] = np.fft.irfft(Tf)
    # return T
    return np.fft.irfft(Tf)

def get_data(var,it,axis,i):
    if put_in_RAM:
        if floatype == np.float16:
            if axis==0:
                return np.float32(data5D[var,it,i,:,:]).T
            elif axis==1:
                return np.float32(data5D[var,it,:,i,:])
            elif axis==2:
                return np.float32(data5D[var,it,:,:,i])
        else:
            if axis==0:
                return data5D[var,it,i,:,:].T
            elif axis==1:
                return data5D[var,it,:,i,:]
            elif axis==2:
                return data5D[var,it,:,:,i]
    else:
        if var==0:
            if axis==0:
                return list_f[it]['WT'][0,nz1+i,nx1:nx2,ny1:ny2].data.T
            elif axis==1:
                return list_f[it]['WT'][0,nz1:nz2,nx1+i,ny1:ny2].data
            elif axis==2:
                return list_f[it]['WT'][0,nz1:nz2,nx1:nx2,ny1+i].data
        if var==1:
            if axis==0:
                data = list_f[it]['THT'][0,nz1+i,nx1:nx2,ny1:ny2].data.T * ( 1+0.61*list_f[it]['RVT'][0,nz1+i,nx1:nx2,ny1:ny2].data.T )
                return data-np.mean(data,axis=(0,1),keepdims=True) # to be replaced by 000.nc mean data
                # return data-np.median(data,axis=(0,1),keepdims=True) # to be replaced by 000.nc mean data
            elif axis==1:
                data = list_f[it]['THT'][0,nz1:nz2,nx1+i,ny1:ny2].data * ( 1+0.61*list_f[it]['RVT'][0,nz1:nz2,nx1+i,ny1:ny2].data )
                return data-np.mean(data,axis=1,keepdims=True) # to be replaced by 000.nc mean data
            elif axis==2:
                data = list_f[it]['THT'][0,nz1:nz2,nx1:nx2,ny1+i].data * ( 1+0.61*list_f[it]['RVT'][0,nz1:nz2,nx1:nx2,ny1+i].data )
                return data-np.mean(data,axis=1,keepdims=True) # to be replaced by 000.nc mean data
            
        if var==2:
            if axis==0:
                data = list_f[it]['RVT'][0,nz1+i,nx1:nx2,ny1:ny2].data.T
            elif axis==1:
                data = list_f[it]['RVT'][0,nz1:nz2,nx1+i,ny1:ny2].data
            elif axis==2:
                data = list_f[it]['RVT'][0,nz1:nz2,nx1:nx2,ny1+i].data
            return data*1000
        if var==3:
            if axis==0:
                data = list_f[it]['RCT'][0,nz1+i,nx1:nx2,ny1:ny2].data.T + list_f[it]['RIT'][0,nz1+i,nx1:nx2,ny1:ny2].data.T
            elif axis==1:
                data = list_f[it]['RCT'][0,nz1:nz2,nx1+i,ny1:ny2].data + list_f[it]['RIT'][0,nz1:nz2,nx1+i,ny1:ny2].data
            elif axis==2:
                data = list_f[it]['RCT'][0,nz1:nz2,nx1:nx2,ny1+i].data + list_f[it]['RIT'][0,nz1:nz2,nx1:nx2,ny1+i].data
            return data*1000
        if var==4:
            if axis==0:
                data = list_f[it]['RRT'][0,nz1+i,nx1:nx2,ny1:ny2].data.T + list_f[it]['RST'][0,nz1+i,nx1:nx2,ny1:ny2].data.T + list_f[it]['RGT'][0,nz1+i,nx1:nx2,ny1:ny2].data.T
            elif axis==1:
                data = list_f[it]['RRT'][0,nz1:nz2,nx1+i,ny1:ny2].data + list_f[it]['RST'][0,nz1:nz2,nx1+i,ny1:ny2].data + list_f[it]['RGT'][0,nz1:nz2,nx1+i,ny1:ny2].data
            elif axis==2:
                data = list_f[it]['RRT'][0,nz1:nz2,nx1:nx2,ny1+i].data + list_f[it]['RST'][0,nz1:nz2,nx1:nx2,ny1+i].data + list_f[it]['RGT'][0,nz1:nz2,nx1:nx2,ny1+i].data
            return data*1000
        
        if var==5:
            if axis==0:
                data = list_f[it]['PABST'][0,nz1+i,nx1:nx2,ny1:ny2].data.T
                return data-np.mean(data,axis=(0,1),keepdims=True) # to be replaced by 000.nc mean data
            elif axis==1:
                data = list_f[it]['PABST'][0,nz1:nz2,nx1+i,ny1:ny2].data
                return data-np.mean(data,axis=1,keepdims=True) # to be replaced by 000.nc mean data
            elif axis==2:
                data = list_f[it]['PABST'][0,nz1:nz2,nx1:nx2,ny1+i].data
                return data-np.mean(data,axis=1,keepdims=True) # to be replaced by 000.nc mean data
            
        if var==6:
            if axis==0:
                return list_f[it]['UT'][0,nz1+i,nx1:nx2,ny1:ny2].data.T
            elif axis==1:
                return list_f[it]['UT'][0,nz1:nz2,nx1+i,ny1:ny2].data
            elif axis==2:
                return list_f[it]['UT'][0,nz1:nz2,nx1:nx2,ny1+i].data
        if var==7:
            if axis==0:
                return list_f[it]['VT'][0,nz1+i,nx1:nx2,ny1:ny2].data.T
            elif axis==1:
                return list_f[it]['VT'][0,nz1:nz2,nx1+i,ny1:ny2].data
            elif axis==2:
                return list_f[it]['VT'][0,nz1:nz2,nx1:nx2,ny1+i].data
            

def regrid_mean(a,n):
    return as_strided(a, shape= tuple(map(lambda x: int(x / n), a.shape)) + (n, n), strides=tuple(map(lambda x: x*n, a.strides)) + a.strides).mean(axis=(2,3))

def get_precip_plot(precip,X,Y,step,threshold=10**-2):
    flat_precip = np.ndarray.flatten(regrid_mean(precip,step))
    select_precip = flat_precip > threshold
    flat_precip = flat_precip[select_precip]
    flat_X = np.ndarray.flatten(regrid_mean(X,step))[select_precip]
    flat_Y = np.ndarray.flatten(regrid_mean(Y,step))[select_precip]
    return flat_X, flat_Y, flat_precip

# Zxx,Zyy = np.meshgrid(x,y)
X,Y = np.meshgrid(x,y)
Yxx,Yzz = np.meshgrid(x,z)
Xyy,Xzz = np.meshgrid(y,z)

def get_hexagonal_flat_coords(x,y,spacing=3,mirror=0):
    ddot = spacing*(x[1]-x[0])
    xf = np.arange(x[0],x[-1]-ddot/2,ddot)
    yf = np.arange(y[0],y[-1],ddot*np.sqrt(3)/2)
    flat_X,flat_Y = np.meshgrid(xf,yf)
    flat_X[mirror::2,:] += ddot/2
    return flat_X.ravel(),flat_Y.ravel()

def get_all_hexagonal_flat_coords(x,y,z,spacings=[5,3,3],mirror=0):
    flat_Zxx,flat_Zyy = get_hexagonal_flat_coords(x,y,spacing=spacings[0],mirror=mirror)
    flat_Yxx,flat_Yzz = get_hexagonal_flat_coords(x,z,spacing=spacings[1],mirror=mirror)
    flat_Xyy,flat_Xzz = get_hexagonal_flat_coords(y,z,spacing=spacings[2],mirror=mirror)
    return flat_Zxx,flat_Zyy,flat_Yxx,flat_Yzz,flat_Xyy,flat_Xzz
flat_Zxx,flat_Zyy,flat_Yxx,flat_Yzz,flat_Xyy,flat_Xzz = get_all_hexagonal_flat_coords(x,y,z,spacings=[nx/70,nx/100,nx/100])
qv_Zxx,qv_Zyy,qv_Yxx,qv_Yzz,qv_Xyy,qv_Xzz = get_all_hexagonal_flat_coords(x,y,z,spacings=[nx/70,nx/100,nx/100],mirror=1)
@njit(parallel=True)
def get_hexagonal_flat_precip(x,y,flat_X,flat_Y,precip):
    n_flat = len(flat_X)
    flat_precip = np.zeros(n_flat)
    for i in prange(n_flat):
        ix = np.argmin(np.abs(x-flat_X[i]))
        iy = np.argmin(np.abs(y-flat_Y[i]))
        flat_precip[i] = precip[iy,ix]
    return flat_precip

#%%


class Player(FuncAnimation):
    def __init__(self,  pv,var_init=0, frames=None, init_func=None, fargs=None,save_count=None,
                 clouds_levels=[1e-3],clouds_color='k',clouds_lw=1.,clouds_alpha=0.7,
                 precip_color='k',precip_alpha=1.,
                 linestyles=['--','-.',':'],line_color='grey',
                 animated=False,blit=False,fs=10,
                 frame_on=True,ticks=True, vert_height=0.4 ,width=19.2, **kwargs):
        self.it = 0
        self.iz = 50
        self.ix = nx//2
        self.iy = ny//2
        
        self.min=0
        self.max=nt-1
        self.runs = True
        self.forwards = True
        self.interpolation = False
        self.clouds_contour = True
        self.precip_dots = True
        self.precip_step = 5
        self.streamlines = False
        self.quiver = False
        
        vert_fact = vert_height/( (z_[-1]-z_[0])/(x_[-1]-x_[0]) )
        vert_left = 1/(1+vert_fact)
        vert_width = vert_fact/(1+vert_fact)
        height = width/(1+vert_fact)
        self.fig = plt.figure(figsize=(width,height))
        
        self.pv = pv
        self.var=var_init
        if ticks:
            pad_down = 0.06
            pad_left = pad_down*height/width
            self.ax_Z = self.fig.add_axes([pad_left, pad_down, vert_left-pad_left, 1-pad_down], frame_on=frame_on)
            self.ax_X = self.fig.add_axes([vert_left+pad_left, pad_down, vert_width-pad_left, vert_height-pad_down],frame_on=frame_on)
            self.ax_Y = self.fig.add_axes([vert_left+pad_left, vert_height+pad_down, vert_width-pad_left, vert_height-pad_down],frame_on=frame_on)
        else:
            self.ax_Z = self.fig.add_axes([0, 0, vert_left, 1],  xticks=[],yticks=[],frame_on=frame_on)
            self.ax_X = self.fig.add_axes([vert_left, 0, vert_width, vert_height],  xticks=[],yticks=[],frame_on=frame_on)
            self.ax_Y = self.fig.add_axes([vert_left, vert_height, vert_width, vert_height],  xticks=[],yticks=[],frame_on=frame_on)
        self.ax_Z.set_xlim([x_[0],x_[-1]])
        self.ax_Z.set_ylim([y_[0],y_[-1]])
        self.ax_X.set_xlim([y_[0],y_[-1]])
        self.ax_X.set_ylim([z_[0],z_[-1]])
        self.ax_Y.set_xlim([x_[0],x_[-1]])
        self.ax_Y.set_ylim([z_[0],z_[-1]])
        self.ax_Z.set_aspect('equal')
        self.ax_X.set_aspect('equal')
        self.ax_Y.set_aspect('equal')
            
        self.line_ZX = self.ax_Z.axvline(x[self.ix],color=line_color,ls=linestyles[1])
        self.line_ZY = self.ax_Z.axhline(y[self.iy],color=line_color,ls=linestyles[2])
        self.line_XY = self.ax_X.axvline(y[self.iy],color=line_color,ls=linestyles[2])
        self.line_XZ = self.ax_X.axhline(z[self.iz],color=line_color,ls=linestyles[0])
        self.line_YX = self.ax_Y.axvline(x[self.ix],color=line_color,ls=linestyles[1])
        self.line_YZ = self.ax_Y.axhline(z[self.iz],color=line_color,ls=linestyles[0])
        
        legend_elements = [ [Line2D([0], [0], color=line_color,linestyle=linestyles[i], lw=1, label=lb+' plane')] for i,lb in enumerate(["X-Y","Y-Z","X-Z"]) ]
        self.ax_Z.legend(handles=legend_elements[0],fancybox=True,fontsize=fs,loc='upper left')#,bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
        self.ax_X.legend(handles=legend_elements[1],fancybox=True,fontsize=fs,loc='upper left')#,bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
        self.ax_Y.legend(handles=legend_elements[2],fancybox=True,fontsize=fs,loc='upper left')#,bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
        
        if pv[var_init][5]=='lin':
            norm=matplotlib.colors.Normalize(vmin=pv[var_init][1], vmax=pv[var_init][2])
        elif pv[var_init][5]=='asinh':
            norm=matplotlib.colors.AsinhNorm(linear_width=pv[var_init][6],vmin=pv[var_init][1], vmax=pv[var_init][2])
        self.image_Z = self.ax_Z.imshow( get_data(self.var,self.it,0,self.iz) ,norm=norm, cmap=pv[var_init][3], extent=(x_[0],x_[-1],y_[0],y_[-1]), origin= 'lower', animated=animated,interpolation='none',zorder=1)
        self.image_X = self.ax_X.pcolormesh(y_,z_, get_data(self.var,self.it,1,self.ix) ,norm=norm, cmap=pv[var_init][3], animated=animated,shading='flat',zorder=1)
        self.image_Y = self.ax_Y.pcolormesh(x_,z_, get_data(self.var,self.it,2,self.iy) ,norm=norm, cmap=pv[var_init][3], animated=animated,shading='flat',zorder=1)

        
        self.clouds_levels,self.clouds_color,self.clouds_lw,self.clouds_alpha = clouds_levels,clouds_color,clouds_lw,clouds_alpha
        self.clouds = [self.ax_Z.contour(x,y,get_data(3,self.it,0,self.iz), levels=clouds_levels,colors=clouds_color,linewidths=clouds_lw,alpha=clouds_alpha),
                       self.ax_X.contour(y,z,get_data(3,self.it,1,self.ix), levels=clouds_levels,colors=clouds_color,linewidths=clouds_lw,alpha=clouds_alpha),
                       self.ax_Y.contour(x,z,get_data(3,self.it,2,self.iy), levels=clouds_levels,colors=clouds_color,linewidths=clouds_lw,alpha=clouds_alpha)]
        
        self.precip_color,self.precip_alpha = precip_color,precip_alpha
        
        
        flat_precip = get_hexagonal_flat_precip(x,y,flat_Zxx,flat_Zyy,get_data(4,self.it,0,self.iz))
        select_precip = flat_precip > 10**-2
        self.precip_Z = self.ax_Z.scatter(flat_Zxx[select_precip], flat_Zyy[select_precip], s= flat_precip[select_precip]*2, c=self.precip_color,alpha=self.precip_alpha,zorder=100)
        flat_precip = get_hexagonal_flat_precip(y,z,flat_Xyy,flat_Xzz,get_data(4,self.it,1,self.ix))
        select_precip = flat_precip > 10**-2
        self.precip_X = self.ax_X.scatter(flat_Xyy[select_precip], flat_Xzz[select_precip], s= flat_precip[select_precip]*2, c=self.precip_color,alpha=self.precip_alpha,zorder=100)
        flat_precip = get_hexagonal_flat_precip(x,z,flat_Yxx,flat_Yzz,get_data(4,self.it,2,self.iy))
        select_precip = flat_precip > 10**-2
        self.precip_Y = self.ax_Y.scatter(flat_Yxx[select_precip], flat_Yzz[select_precip], s= flat_precip[select_precip]*2, c=self.precip_color,alpha=self.precip_alpha,zorder=100)
        
        self.stream_Z = None
        
        qvc,qvp,qvs,qvsu,qvu,qvw,qvhw,qvhl ='k','mid',10,'xy','xy',0.1,2,3
        flat_U = get_hexagonal_flat_precip(x,y,qv_Zxx,qv_Zyy,get_data(6,self.it,0,self.iz))
        flat_V = get_hexagonal_flat_precip(x,y,qv_Zxx,qv_Zyy,get_data(7,self.it,0,self.iz))
        self.quiver_Z = self.ax_Z.quiver(qv_Zxx,qv_Zyy,flat_V,flat_U, color=qvc,pivot=qvp,scale=qvs,scale_units=qvsu,units=qvu,width=qvw,headwidth=qvhw,headlength=qvhl,visible=False,zorder=2)
        flat_U = get_hexagonal_flat_precip(y,z,qv_Xyy,qv_Xzz,get_data(0,self.it,1,self.ix))
        flat_V = get_hexagonal_flat_precip(y,z,qv_Xyy,qv_Xzz,get_data(6,self.it,1,self.ix))
        self.quiver_X = self.ax_X.quiver(qv_Xyy,qv_Xzz,flat_V,flat_U, color=qvc,pivot=qvp,scale=qvs,scale_units=qvsu,units=qvu,width=qvw,headwidth=qvhw,headlength=qvhl,visible=False,zorder=2)
        flat_U = get_hexagonal_flat_precip(x,z,qv_Yxx,qv_Yzz,get_data(0,self.it,2,self.iy))
        flat_V = get_hexagonal_flat_precip(x,z,qv_Yxx,qv_Yzz,get_data(7,self.it,2,self.iy))
        self.quiver_Y = self.ax_Y.quiver(qv_Yxx,qv_Yzz,flat_V,flat_U, color=qvc,pivot=qvp,scale=qvs,scale_units=qvsu,units=qvu,width=qvw,headwidth=qvhw,headlength=qvhl,visible=False,zorder=2)
                
        pos= [vert_left + vert_width/10 , 0.955 , 8/10*vert_width , 0.04]
        self.playerax = self.fig.add_axes(pos,  xticks=[],yticks=[],frame_on=frame_on)
        self.setup(self.playerax , pv)
        
        self.filterax = self.fig.add_axes([pos[0]+pos[2]/3,pos[1]-1*pos[3] ,pos[2]/3, pos[3] ],  xticks=[],yticks=[],frame_on=frame_on)
        self.lamda_filter = 0.
        self.filter_slider = matplotlib.widgets.Slider(self.filterax, 'Filter min wavelength = ',0., 6, valinit=0.,orientation ='horizontal' ,color='darkorange')
        self.filter_text = self.filterax.text(0.5,0.5,'0 km', transform=self.filterax.transAxes,ha='center',va='center')
        self.filter_slider.on_changed(self.set_lamda_filter)
        
        self.ax_CB = self.fig.add_axes([pos[0],pos[1]-2.2*pos[3] ,pos[2], pos[3] ],  xticks=[],yticks=[],frame_on=frame_on)
        self.CB = plt.colorbar(self.image_Z,cax=self.ax_CB,orientation='horizontal',format='%.1f')
        self.CB_fact = 1.5
        self.CB.ax.tick_params(labelsize=fs)
        self.CB.set_label(self.pv[self.var][0]+'  ( '+self.pv[self.var][4]+' )',fontsize=1.5*fs)
#        self.time_text = ax_profil.text(Tmax*1/10,zmax,"Ra = %.2e"%les_Ra[0],fontsize=18)
        self.fig.canvas.mpl_connect('button_press_event',self.on_press)
        
        FuncAnimation.__init__(self,self.fig, self.update, frames=self.play(), 
                                           init_func=init_func, fargs=fargs,blit=blit,
                                           save_count=save_count, **kwargs )
        
    def interpolation_update(self):
        if self.pv[self.var][5]=='lin':
            norm=matplotlib.colors.Normalize(vmin=self.pv[self.var][1], vmax=self.pv[self.var][2])
        elif self.pv[self.var][5]=='asinh':
            norm=matplotlib.colors.AsinhNorm(linear_width=self.pv[self.var][6],vmin=self.pv[self.var][1], vmax=self.pv[self.var][2])
        if self.interpolation==True:
            self.image_Z.set_interpolation('bicubic')
            self.image_X.remove() ; self.image_X = self.ax_X.pcolormesh(y,z,get_data(self.var,self.it,1,self.ix) ,norm=norm, cmap=self.pv[self.var][3],shading='gouraud',zorder=1)
            self.image_Y.remove() ; self.image_Y = self.ax_Y.pcolormesh(x,z,get_data(self.var,self.it,2,self.iy) ,norm=norm, cmap=self.pv[self.var][3],shading='gouraud',zorder=1)
        else:
            self.image_Z.set_interpolation('none')
            self.image_X.remove() ; self.image_X = self.ax_X.pcolormesh(y_,z_,get_data(self.var,self.it,1,self.ix) ,norm=norm, cmap=self.pv[self.var][3],shading='flat',zorder=1)
            self.image_Y.remove() ; self.image_Y = self.ax_Y.pcolormesh(x_,z_,get_data(self.var,self.it,2,self.iy) ,norm=norm, cmap=self.pv[self.var][3],shading='flat',zorder=1)
        self.fig.canvas.draw_idle()
    
    def remove_clouds(self):
        if self.clouds is not None:
            for i in range(3):
                for tp in self.clouds[i].collections:
                    tp.remove()
            self.clouds = None
        
    def remove_precip(self):
        if self.precip_Z is not None:
            self.precip_Z.remove()
            self.precip_X.remove()
            self.precip_Y.remove()
            self.precip_Z = None
    def remove_streamlines(self):
        if self.stream_Z is not None:
            self.stream_Z.lines.remove()
            self.stream_X.lines.remove()
            self.stream_Y.lines.remove()
            for ax in [self.ax_Z,self.ax_X,self.ax_Y]:
                for art in ax.get_children():
                    if isinstance(art,matplotlib.patches.FancyArrowPatch):
                        art.remove()
            self.stream_Z = None
            
    def data_update(self):
        if self.lamda_filter>0.:
            self.image_Z.set_data( gaussian_filter(get_data(self.var,self.it,0,self.iz),self.lamda_filter) )
            self.image_X.set_array( np.ravel(gaussian_sp_fft_tridiag_filter(get_data(self.var,self.it,1,self.ix),self.lamda_filter),order='C') )
            self.image_Y.set_array( np.ravel(gaussian_sp_fft_tridiag_filter(get_data(self.var,self.it,2,self.iy),self.lamda_filter),order='C') )
        else:
            self.image_Z.set_data( get_data(self.var,self.it,0,self.iz) )
            self.image_X.set_array( np.ravel(get_data(self.var,self.it,1,self.ix),order='C') )
            self.image_Y.set_array( np.ravel(get_data(self.var,self.it,2,self.iy),order='C') )
        
        if self.clouds_contour:
            self.remove_clouds()
            self.clouds = [self.ax_Z.contour(x,y,get_data(3,self.it,0,self.iz), levels=self.clouds_levels,colors=self.clouds_color,linewidths=self.clouds_lw,alpha=self.clouds_alpha),
                          self.ax_X.contour(y,z,get_data(3,self.it,1,self.ix), levels=self.clouds_levels,colors=self.clouds_color,linewidths=self.clouds_lw,alpha=self.clouds_alpha),
                          self.ax_Y.contour(x,z,get_data(3,self.it,2,self.iy), levels=self.clouds_levels,colors=self.clouds_color,linewidths=self.clouds_lw,alpha=self.clouds_alpha)]
            
        if self.precip_dots:
            self.remove_precip()
            flat_precip = get_hexagonal_flat_precip(x,y,flat_Zxx,flat_Zyy,get_data(4,self.it,0,self.iz))
            select_precip = flat_precip > 10**-2
            self.precip_Z = self.ax_Z.scatter(flat_Zxx[select_precip], flat_Zyy[select_precip], s= flat_precip[select_precip]*2, c=self.precip_color,alpha=self.precip_alpha,zorder=100)
            flat_precip = get_hexagonal_flat_precip(y,z,flat_Xyy,flat_Xzz,get_data(4,self.it,1,self.ix))
            select_precip = flat_precip > 10**-2
            self.precip_X = self.ax_X.scatter(flat_Xyy[select_precip], flat_Xzz[select_precip], s= flat_precip[select_precip]*2, c=self.precip_color,alpha=self.precip_alpha,zorder=100)
            flat_precip = get_hexagonal_flat_precip(x,z,flat_Yxx,flat_Yzz,get_data(4,self.it,2,self.iy))
            select_precip = flat_precip > 10**-2
            self.precip_Y = self.ax_Y.scatter(flat_Yxx[select_precip], flat_Yzz[select_precip], s= flat_precip[select_precip]*2, c=self.precip_color,alpha=self.precip_alpha,zorder=100)
            
        if self.streamlines:
            self.remove_streamlines()
            U = get_data(6,self.it,0,self.iz) ; V = get_data(7,self.it,0,self.iz)
            self.stream_Z = self.ax_Z.streamplot(x, y,V,U,density=1., color='k', linewidth=np.sqrt(U**2+V**2)/10) #
            U = interp_vertical(get_data(6,self.it,1,self.ix),ios,c0s,c1s) ; V = interp_vertical(get_data(0,self.it,1,self.ix),ios,c0s,c1s)
            self.stream_X = self.ax_X.streamplot(y, new_z, U,V,density=2., color='k', linewidth=np.sqrt(U**2+V**2)/10)
            U = interp_vertical(get_data(7,self.it,2,self.iy),ios,c0s,c1s) ; V = interp_vertical(get_data(0,self.it,2,self.iy),ios,c0s,c1s)
            self.stream_Y = self.ax_Y.streamplot(x, new_z, U,V ,density=2., color='k', linewidth=np.sqrt(U**2+V**2)/10)
            
        if self.quiver:
            flat_U = get_hexagonal_flat_precip(x,y,qv_Zxx,qv_Zyy,get_data(6,self.it,0,self.iz))
            flat_V = get_hexagonal_flat_precip(x,y,qv_Zxx,qv_Zyy,get_data(7,self.it,0,self.iz))
            self.quiver_Z.set_UVC(flat_V,flat_U)
            flat_U = get_hexagonal_flat_precip(y,z,qv_Xyy,qv_Xzz,get_data(0,self.it,1,self.ix))
            flat_V = get_hexagonal_flat_precip(y,z,qv_Xyy,qv_Xzz,get_data(6,self.it,1,self.ix))
            self.quiver_X.set_UVC(flat_V,flat_U)
            flat_U = get_hexagonal_flat_precip(x,z,qv_Yxx,qv_Yzz,get_data(0,self.it,2,self.iy))
            flat_V = get_hexagonal_flat_precip(x,z,qv_Yxx,qv_Yzz,get_data(7,self.it,2,self.iy))
            self.quiver_Y.set_UVC(flat_V,flat_U)
        self.fig.canvas.draw_idle()

    def cmap_update(self):
        if self.pv[self.var][5]=='lin':
            norm=matplotlib.colors.Normalize(vmin=self.pv[self.var][1], vmax=self.pv[self.var][2])
        elif self.pv[self.var][5]=='asinh':
            norm=matplotlib.colors.AsinhNorm(linear_width=self.pv[self.var][6],vmin=self.pv[self.var][1], vmax=self.pv[self.var][2])
        self.image_Z.set_norm(norm)
        self.image_X.set_norm(norm)
        self.image_Y.set_norm(norm)
        self.image_Z.set_cmap(self.pv[self.var][3])
        self.image_X.set_cmap(self.pv[self.var][3])
        self.image_Y.set_cmap(self.pv[self.var][3])
        self.CB.set_label(self.pv[self.var][0]+'  '+self.pv[self.var][4])
        
    def play(self):
        while self.runs:
            if self.forwards:
                self.it+=1
            else:
                self.it-=1
            if self.it > self.min and self.it < self.max:
                yield self.it
            else:
                self.stop()
                yield self.it

    def start(self):
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()
    def backward(self, event=None):
        self.forwards = False
        self.start()
    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()
    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.it > self.min and self.it < self.max:
            self.it = self.it+self.forwards-(not self.forwards)
        elif self.it == self.min and self.forwards:
            self.it+=1
        elif self.it == self.max and not self.forwards:
            self.it-=1
        self.data_update()
        self.slider.set_val(self.it)
        self.fig.canvas.draw_idle()

    def var0(self,event=None):
        self.var = 0 ; self.cmap_update() ; self.data_update()
    def var1(self,event=None):
        self.var = 1 ; self.cmap_update() ; self.data_update()
    def var2(self,event=None):
        self.var = 2 ; self.cmap_update() ; self.data_update()
    def var3(self,event=None):
        self.var = 3 ; self.cmap_update() ; self.data_update()
    def var4(self,event=None):
        self.var = 4 ; self.cmap_update() ; self.data_update()
    def var5(self,event=None):
        self.var = 5 ; self.cmap_update() ; self.data_update()
    def var6(self,event=None):
        self.var = 6 ; self.cmap_update() ; self.data_update()
    def var7(self,event=None):
        self.var = 7 ; self.cmap_update() ; self.data_update()
    def setup(self, playerax,pv):
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        self.sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(playerax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(self.sliderax, '', self.min, self.max, valinit=self.it, color='deepskyblue')
        self.time_text = self.sliderax.text(0.5,0.5, times[self.it], transform=self.sliderax.transAxes,ha='center',va='center')
        self.slider.on_changed(self.set_pos)

        for v,pvar in enumerate(pv):
            ax = divider.append_axes("right", size="80%", pad=0.05)
            if v==0:
                self.button0 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button0.on_clicked(self.var0)
            if v==1:
                self.button1 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button1.on_clicked(self.var1)
            if v==2:
                self.button2 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2.on_clicked(self.var2)
            if v==3:
                self.button3 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button3.on_clicked(self.var3)
            if v==4:
                self.button4 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button4.on_clicked(self.var4)
            if v==5:
                self.button5 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button5.on_clicked(self.var5)
            if v==6:
                self.button6 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button6.on_clicked(self.var6)
            if v==7:
                self.button7 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button7.on_clicked(self.var7)
                
    def set_pos(self,i):
        self.it = int(i)#int(self.slider.val)
        self.data_update()
        self.time_text.set_text(times[self.it])
    def update(self,i):
        self.slider.set_val(i)
    
    def set_lamda_filter(self,l):
        self.lamda_filter = l**3
        if self.lamda_filter<1:
            self.filter_text.set_text('{:.2f} km'.format(self.lamda_filter))
        elif self.lamda_filter<10:
            self.filter_text.set_text('{:.1f} km'.format(self.lamda_filter))
        else:
            self.filter_text.set_text('{:.0f} km'.format(self.lamda_filter))
        self.data_update()
        
    def on_press(self,event):
        if event.inaxes == self.ax_Z:
            if event.button == 3:
                self.iy = np.argmin(np.abs(y - event.ydata)) ; self.line_ZY.set_ydata(y[self.iy]) ; self.line_XY.set_xdata(y[self.iy]) ; self.data_update() ; self.fig.canvas.draw_idle()
            if event.button == 1:
                self.ix = np.argmin(np.abs(x - event.xdata)) ; self.line_ZX.set_xdata(x[self.ix]) ; self.line_YX.set_xdata(x[self.ix]) ; self.data_update() ; self.fig.canvas.draw_idle()
        if event.inaxes == self.ax_X:
            if event.button == 3:
                self.iz = np.argmin(np.abs(z - event.ydata)) ; self.line_XZ.set_ydata(z[self.iz]) ; self.line_YZ.set_ydata(z[self.iz]) ; self.data_update() ; self.fig.canvas.draw_idle()
            if event.button == 1:
                self.iy = np.argmin(np.abs(y - event.xdata)) ; self.line_XY.set_xdata(y[self.iy]) ; self.line_ZY.set_ydata(y[self.iy]) ; self.data_update() ; self.fig.canvas.draw_idle()
        if event.inaxes == self.ax_Y:
            if event.button == 3:
                self.iz = np.argmin(np.abs(z - event.ydata)) ; self.line_YZ.set_ydata(z[self.iz]) ; self.line_XZ.set_ydata(z[self.iz]) ; self.data_update() ; self.fig.canvas.draw_idle()
            if event.button == 1:
                self.ix = np.argmin(np.abs(x - event.xdata)) ; self.line_YX.set_xdata(x[self.ix]) ; self.line_ZX.set_xdata(x[self.ix]) ; self.data_update() ; self.fig.canvas.draw_idle()
                
        if event.inaxes == self.ax_CB:
            if event.button == 3:
                self.pv[self.var][1] *= self.CB_fact
                self.pv[self.var][2] *= self.CB_fact
                self.cmap_update() ; self.fig.canvas.draw_idle()
            if event.button == 1:
                self.pv[self.var][1] /= self.CB_fact
                self.pv[self.var][2] /= self.CB_fact
                self.cmap_update() ; self.fig.canvas.draw_idle()
        
# Wcm = mcolors.LinearSegmentedColormap.from_list('seismic2',['k','b','c','w','y','r','k'])
# Wcm = mcolors.LinearSegmentedColormap.from_list('seismic2',[(0.2,0.2,0.2),(0,0,1),(0,1,1),(1,1,1),(1,1,0),(1,0,0),(0.2,0.2,0.2)])
Wcm = mcolors.LinearSegmentedColormap.from_list('seismic2',[(0,(0.2,0.2,0.2)),(0.25,(0,0,1)),(0.4,(0,1,1)),(0.5,(1,1,1)),(0.6,(1,1,0)),(0.75,(1,0,0)),(1,(0.2,0.2,0.2))])
Tcm = 'RdBu_r'
Rcm = 'YlGnBu'
RHcm = 'rainbow_r'
parametres_variables = [['w',-40,40,Wcm,'m/s','asinh',5],['$θ_v^{\prime}$',-6,6, Tcm,'K','asinh',0.15]]
if relative_humidity:
    parametres_variables += [['$RH$',0,100,RHcm,'%','lin']]
else:
    parametres_variables += [['$r_v$',0,20,Rcm,'g/kg','lin']]

parametres_variables += [ ['$r_c$',0,2,Rcm,'g/kg','lin'],['$r_p$',0,2,Rcm,'g/kg','lin'] ,['$p^{\prime}$',-100,100,'seismic','Pa','asinh',5]]
if wind:
    parametres_variables += [['u',-40,40,Wcm,'m/s','asinh',5],['v',-40,40,Wcm,'m/s','asinh',5]]
ani = Player(parametres_variables,interval=10,ticks=True)

class Interpolation_tool(ToolToggleBase):
    # default_keymap = 'control'
    description = 'Turn on/off interpolation ( bicubic for X-Y & gouraud for X-Z and Y-Z )'
    default_toggled = False
    def __init__(self,*args, ani, **kwargs):
        self.ani = ani
        super().__init__(*args, **kwargs)
    def enable(self,*args):
        self.ani.interpolation= True
        self.ani.interpolation_update()
    def disable(self, *args):
        self.ani.interpolation= False
        self.ani.interpolation_update()
ani.fig.canvas.manager.toolmanager.add_tool('Interpolation', Interpolation_tool,ani=ani)
ani.fig.canvas.manager.toolbar.add_tool('Interpolation', 'animation')

class Clouds_tool(ToolToggleBase):
    # default_keymap = 'control'
    description = 'Turn on/off clouds contour ( 0.001 g/kg )'
    default_toggled = True
    def __init__(self,*args, ani, **kwargs):
        self.ani = ani
        super().__init__(*args, **kwargs)
    def enable(self,*args):
        self.ani.clouds_contour = True
        self.ani.data_update()
    def disable(self, *args):
        self.ani.clouds_contour = False
        self.ani.remove_clouds()
        self.ani.fig.canvas.draw_idle()
ani.fig.canvas.manager.toolmanager.add_tool('Clouds', Clouds_tool,ani=ani)
ani.fig.canvas.manager.toolbar.add_tool('Clouds', 'animation')

class Precip_tool(ToolToggleBase):
    # default_keymap = 'control'
    description = 'Turn on/off precipitation dots ( > 0.1 g/kg )'
    default_toggled = True
    def __init__(self,*args, ani, **kwargs):
        self.ani = ani
        super().__init__(*args, **kwargs)
    def enable(self,*args):
        self.ani.precip_dots = True
        self.ani.data_update()
    def disable(self, *args):
        self.ani.precip_dots = False
        self.ani.remove_precip()
        self.ani.fig.canvas.draw_idle()
ani.fig.canvas.manager.toolmanager.add_tool('Precipitations', Precip_tool,ani=ani)
ani.fig.canvas.manager.toolbar.add_tool('Precipitations', 'animation')

if wind:
    class Streamlines_tool(ToolToggleBase):
        # default_keymap = 'control'
        description = 'Turn on/off streamlines ( m/s )'
        default_toggled = False
        def __init__(self,*args, ani, **kwargs):
            self.ani = ani
            super().__init__(*args, **kwargs)
        def enable(self,*args):
            self.ani.streamlines = True
            self.ani.data_update()
        def disable(self, *args):
            self.ani.streamlines = False
            self.ani.remove_streamlines()
            self.ani.fig.canvas.draw_idle()
    ani.fig.canvas.manager.toolmanager.add_tool('Streamlines', Streamlines_tool,ani=ani)
    ani.fig.canvas.manager.toolbar.add_tool('Streamlines', 'animation')
    
    class Quiver_tool(ToolToggleBase):
        # default_keymap = 'control'
        description = 'Turn on/off Quiver ( m/s )'
        default_toggled = False
        def __init__(self,*args, ani, **kwargs):
            self.ani = ani
            super().__init__(*args, **kwargs)
        def enable(self,*args):
            self.ani.quiver = True
            self.ani.quiver_Z.set_visible(True)
            self.ani.quiver_Y.set_visible(True)
            self.ani.quiver_X.set_visible(True)
            self.ani.data_update()
        def disable(self, *args):
            self.ani.quiver = False
            self.ani.quiver_Z.set_visible(False)
            self.ani.quiver_Y.set_visible(False)
            self.ani.quiver_X.set_visible(False)
            self.ani.fig.canvas.draw_idle()
    ani.fig.canvas.manager.toolmanager.add_tool('Quiver', Quiver_tool,ani=ani)
    ani.fig.canvas.manager.toolbar.add_tool('Quiver', 'animation')
    