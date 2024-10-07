# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 01:50:07 2022

@author: Nathan
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit,prange
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import xarray as xr
class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 mini=0, maxi=100, pos=(0.125, 0.92), **kwargs):
        self.i = 0
        self.min=mini
        self.max=maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self,self.fig, self.update, frames=self.play(), 
                                           init_func=init_func, fargs=fargs,
                                           save_count=self.max, **kwargs )    

    def play(self):
        while self.runs:
            self.i = self.i+self.forwards-(not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

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
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i+=1
        elif self.i == self.max and not self.forwards:
            self.i-=1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0],pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="500%", pad=0.07)
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
        self.slider = matplotlib.widgets.Slider(sliderax, '', 
                                                self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self,i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self,i):
        self.slider.set_val(i)
        
#%%
Rd = 287.04     # J/kg/K 
g = 9.81
Patm = 101325
Cp = Rd*7/2
Rv = 461 # J/kg/K
Lv =  2.5e6 # J/kg

def create_z(dz0,dzmax,zmax,stretching = 1.1):
    z = 0.
    dz = dz0
    list_z = []
    while z < zmax:
        list_z.append(z)
        z += dz
        dz = min(dzmax,dz*stretching)
    return np.array(list_z)

def P_Rho_Theta_mean(T,z):
    n=len(T)
    P = Patm*np.ones(n)
    integral = 0.
    for i in range(1,n):
        integral += 2*(z[i]-z[i-1])/(T[i]+T[i-1])
        P[i] *= np.exp(-g/Rd*integral)
    return P , P/Rd/T , T*(Patm/P)**(Rd/Cp)

#%%
@njit()
def TDMAsolver(a, b, c, d):
    n = len(d)
    for i in range(n-1):
        m = a[i]/b[i]
        b[i+1] -= m*c[i] 
        d[i+1] -= m*d[i]
    b[-1] = d[-1]/b[-1]
    for i in range(n-2, -1, -1):
        b[i] = (d[i]-c[i]*b[i+1])/b[i]
    return b
            
@njit()
def iter_ED_TH(dt,z,TH,RHO,zi,H,Wstar):
    nz, = z.shape
    K = 0.4 /6 # karman constant for eddy diffusivity defined as  K*Wstar*z_*np.maximum( 0.,1-z_/zi)**2
    # pre-calculs
    z_ = (z[1:]+z[:-1])/2
    # print(np.max(K*Wstar*(z_+10)*np.maximum( 0.,1-z_/zi)**2))
    RK_dz =  (RHO[1:]+RHO[:-1])/2 * K*Wstar*(z_+10)*np.maximum( 0.,1-z_/zi)**2 / (z[1:]-z[:-1])  # nz-1 # 
    dt_Rdz = np.concatenate(( np.array([dt/ RHO[0]/z_[0]]) , dt/ RHO[1:-1]/(z_[1:]-z_[:-1]) )) # nz-1
    # preparation des diagonales a,b,c et d
    a = -dt_Rdz[1:] * RK_dz[:-1]
    c = -dt_Rdz * RK_dz
    b = 1-c
    b[1:] -= a
    d = TH[:-1]
    d[0] += dt_Rdz[0] * H/Cp
    d[-1] += c[-1] * TH[-1]
    TH[:-1] = TDMAsolver(a, b, c, d)
    EDFlux = -RK_dz * (TH[1:]-TH[:-1])
    return TH,EDFlux*Cp

# @njit()
# def iter_ED_TH_explicite(dt,z,TH,RHO,zi,H,Wstar):
#     nz, = z.shape
#     K = 0.4 # karman constant for eddy diffusivity defined as  K*Wstar*z_*np.maximum( 0.,1-z_/zi)**2
#     # pre-calculs
#     z_ = (z[1:]+z[:-1])/2
#     RK_dz =  (RHO[1:]+RHO[:-1])/2 * K*Wstar/6*z_*np.maximum( 0.,1-z_/zi)**2 / (z[1:]-z[:-1])  # nz-1 # 
#     dt_Rdz = np.concatenate(( np.array([dt/ RHO[0]/z_[0]]) , dt/ RHO[1:-1]/(z_[1:]-z_[:-1]) )) # nz-1
#     # résolution explicite
#     EDFlux = -RK_dz * (TH[1:]-TH[:-1])
#     TH[0] -= dt_Rdz[0] * ( EDFlux[0] - H/Cp )
#     TH[1:-1] -= dt_Rdz[1:] * ( EDFlux[1:]-EDFlux[:-1] )
#     return TH,EDFlux*Cp

@njit()
def diag_zi(z,TH,RHO): # Calcul de zi = altitude de l'inversion = Boundary Layer Height
    nz, = z.shape
    deltaTH = 0.3 # seuil pour la détection de l'inversion
    mean_TH = TH[0]
    mass = 0.
    iz = 0
    while iz < nz-1 and TH[iz] < mean_TH + deltaTH:
        layer_mass = (RHO[iz+1]+RHO[iz])/2 * (z[iz+1]-z[iz])
        mean_TH = ( mean_TH*mass + (TH[iz+1]+TH[iz])/2 * layer_mass ) / ( mass+layer_mass)
        mass += layer_mass
        iz+=1
    if iz > 0:
        a = ( TH[iz] - mean_TH - deltaTH ) / (TH[iz]-TH[iz-1])
        return a*z[iz-1] + (1-a)*z[iz]
    else:
        return 0.
    
@njit()
def param_thermiques_plaine(z,TH,RHO,Wstar,Ht):
    nz, = z.shape
    THt,Wt = [ np.full(nz,np.nan) for i in range(2) ]
    dTHdt,FTH,At,Ft,Et,Dt = [ np.zeros(nz) for i in range(6) ]
    # dTHdt,THt,THe,Wt,At,Ae,Ft,Et,Dt = np.zeros(nz),np.zeros(nz),np.zeros(nz),np.zeros(nz),np.zeros(nz)
    
    # Constantes de la param
    a = 1 # Coefficient de réduction de flottabilité
    Cw0 = 0.2
    At0 = 0.15 # EDKF At0*Cw0 = 0.065
    D0 = 0.001
    
    # Fermeture
    Wt[0] = Cw0 * Wstar
    At[0] = At0 ; Ae = 1-At[0]
    Ft[0] = At[0]*RHO[0]*Wt[0]
    THt[0] = TH[0] + Ae*Ht/Cp/Ft[0]
    FTH[0] = Ht/Cp
    
    # Evolution verticale
    iz = 0 ; jz = 1
    while jz < nz:
        dz = z[jz]-z[iz]
        Bt = a * g * (THt[iz]-TH[iz])/Ae/TH[iz]
        if Bt>0:
            Et[iz] = Ae/(1+Ae)*Bt/Wt[iz]**2
            Dt[iz] = D0 + (RHO[iz]-RHO[jz])/dz/RHO[iz]
        else:
            Dt[iz] = -10*Bt/Wt[iz]**2 + D0 + (RHO[iz]-RHO[jz])/dz/RHO[iz]
        
        Wt[jz] = Wt[iz] + dz* ( -Et[iz]*Wt[iz]/Ae + Bt/Wt[iz] )
        # Ft[jz] = Ft[iz] + dz*Ft[iz]*(Et[iz]-Dt[iz])
        if Wt[jz] <= 0:# or Ft[jz] <= 0:
            dTHdt[iz] = FTH[iz]/dz/RHO[iz]
            break
        else:
            THt[jz] = THt[iz] + dz* Et[iz]*(TH[iz]-THt[iz])/Ae
            Ft[jz] = Ft[iz] * np.exp(dz*(Et[iz]-Dt[iz]))
            
            At[jz] = Ft[jz] / Wt[jz] / RHO[jz]
            if At[jz] > 1:
                print("alpha > 1")
                break
            Ae = 1-At[jz]
            FTH[jz] = Ft[jz] * (THt[jz]-TH[jz])/Ae
            
            dTHdt[iz] = (FTH[iz]-FTH[jz])/dz/RHO[iz]
            
            iz = jz
            jz += 1
            
    return dTHdt, THt,FTH*Cp,Wt,At,Ft,Et,-Dt
#%%  
ED = True
MF = True
LES = True
time_integration = False ; dt =60
if LES:
    simu = 'DFLAT' ; seg = 'M100m' ; zmax = 3000.
    userPath = '/home/philippotn'
    dataPath = userPath+'/Documents/NO_SAVE/'
    # dataPath = userPath+'/Documents/SIMU_LES/'
    savePath = userPath+'/Images/LES_'+simu+'/Plot_param_vs_LES/'
    f = xr.open_dataset(dataPath+simu+'_'+seg+'_mean.nc')
    RHO , TH = f['RHO'].data[0], f['TH'].data[0] ; T0=TH[0]
    H = f['H'].data[:]
    nt = len(f['time'])
    z = f['z'].data ; nz = len(z)
else:
    zmax = 3000. # mètres
    print("Hauteur du domaine (m) = ",zmax)
    # z = np.linspace(0,zmax,nz)
    # z = create_z(20,100,zmax,1.2)
    z = create_z(5,20,zmax,1.2)
    # z = create_z(50,50,zmax,1.2)
    nz = len(z)
    
    T0 = 280 # 287 K = temperature mondiale moyenne au niveau de la mer
    BLH0 = 0000 # m
    iz_BLH0 = np.argmin(np.abs(z-BLH0))
    T = T0 -g/Cp*z # -6.K/km = gradient de température moyen dans la troposhère
    T[iz_BLH0:] = T[iz_BLH0] -6.5/1000*(z[iz_BLH0:]-BLH0) # -6.5K/km = gradient de température moyen dans la troposhère
    
    P , RHO , TH = P_Rho_Theta_mean(T,z)
    
    tmax = 3600*24/2
    dt = 60 # secondes
    time = np.arange(0,tmax,dt)
    nt = len(time) # tmax//dt
    H = 350 * np.maximum(0,np.sin(np.pi*2*time/(3600*24)))

#%%
data_TH = np.zeros((nt,nz))
data_EDF = np.zeros((nt,nz))
data_MFF = np.zeros((nt,nz))
data_THt = np.zeros((nt,nz))
data_Wt = np.zeros((nt,nz))
data_At = np.zeros((nt,nz))
data_Ft = np.zeros((nt,nz))
data_Et = np.zeros((nt,nz))
data_Dt = np.zeros((nt,nz))

zi = np.zeros(nt)
Wstar = np.zeros(nt)

## Boucle temporelle##
for it in range(nt):
    if not(time_integration):
        TH = f['TH'].data[it] ; dt=1.
        
    if H[it]>0:
        zi[it] = diag_zi(z,TH,RHO)
        Wstar[it] = ( g/TH[0] * H[it]/RHO[0]/Cp * zi[it] )**(1/3)
        
        if ED:
            TH,data_EDF[it,:-1] = iter_ED_TH(dt,z,TH,RHO,zi[it],H[it],Wstar[it])
            # TH,data_EDF[it,:-1] = iter_ED_TH_explicite(dt,z,TH,RHO,zi[it],H[it],Wstar[it])
            Ht = 0*H[it]
        else:
            Ht = H[it]
        
        if MF:
            dTHdt_MF,data_THt[it],data_MFF[it],data_Wt[it],data_At[it],data_Ft[it],data_Et[it],data_Dt[it] = param_thermiques_plaine(z,TH,RHO,Wstar[it],Ht)
            TH += dt*dTHdt_MF

    data_TH[it] = TH
#%%
# import matplotlib.animation as animation
ncols = 6
ft=15

fig, axs = plt.subplots(ncols=ncols,figsize=(15,5))
# fig.figsize = (16, 5)

xlims = [[T0-2,T0+10],[-100,400],[0,5],[0,1],[0,1],[-0.1,0.1]]
xlabels = ['$\\theta$ (K)', '$c_p \\rho \\overline{w^{\prime} \\theta^{\prime}}$ (W/m²)', 'W (m/s)', '$\\alpha$' ,'f','$\\epsilon$ and $\\delta$ ($m^{-1}$)']


for i in range(ncols):
    axs[i].set_ylim([0,zmax])
    axs[i].grid(axis='y')
    if i==0:
        axs[i].set_ylabel('z (m)',fontsize=ft)
    else:
        # axs[i].set_yticks([])#,fontsize=0)
        axs[i].set_yticklabels([])
    
    axs[i].set_xlabel(xlabels[i],fontsize=ft)
    axs[i].set_xlim(xlims[i])

axs[0].axvline(x=T0,color='grey',linestyle='-',linewidth=1,label='$\\theta_{ini}$')
line_zi = axs[0].axhline(y=zi[0],color='k',linestyle='--',label='$z_i$')
line_TH, = axs[0].plot([],[],label='$\overline{\\theta}$')
line_THt,= axs[0].plot([],[],label='$\\theta_t$')
line_H = axs[1].axvline(x=H[0],color='k',linestyle='--',label='$H$')
line_EDF,= axs[1].plot([],[],label='$\overline{w^{\prime}\\theta^{\prime}}_{ED}$') 
line_THF,= axs[1].plot([],[],label='$\overline{w^{\prime}\\theta^{\prime}}_{EDMF}$')
line_THF_LES,= axs[1].plot([],[],label='$\overline{w^{\prime}\\theta^{\prime}}^{LES}$')
line_Wstar = axs[2].axvline(x=Wstar[0],color='k',linestyle='--',label='$w_*$')
line_Wt, = axs[2].plot([],[],label='$w_t$') 
line_At, = axs[3].plot([],[],label='$\\alpha_t$')
line_Ft, = axs[4].plot([],[],label='$f_t$') 
line_Et, = axs[5].plot([],[],label='$\\epsilon_t$')
line_Dt, = axs[5].plot([],[],label='$\\delta_t$')

if LES:
    line_TH_LES, = axs[0].plot([],[],label='$\overline{\\theta}^{LES}$')
    line_THt_LES,= axs[0].plot([],[],label='$\\theta_t^{LES}$')
    line_Wt_LES, = axs[2].plot([],[],label='$w_t^{LES}$') 
    line_At_LES, = axs[3].plot([],[],label='$\\alpha_t^{LES}$')

for ax in axs:
    ax.legend()
fig.suptitle("Paramétrisation des thermiques en plaine")

def update_plot(it):
    line_zi.set_data([0,1],[zi[it],zi[it]])
    line_H.set_data([H[it],H[it]],[0,1])
    line_Wstar.set_data([Wstar[it],Wstar[it]],[0,1])
    
    line_TH.set_data(data_TH[it], z)
    line_THt.set_data(data_THt[it], z)
    line_EDF.set_data(data_EDF[it], z)
    line_THF.set_data(data_EDF[it]+data_MFF[it], z)
    line_Wt.set_data(data_Wt[it], z)
    line_At.set_data(data_At[it], z)
    line_Ft.set_data(data_Ft[it], z)
    line_Et.set_data(data_Et[it], z)
    line_Dt.set_data(data_Dt[it], z)
    if LES:
        line_TH_LES.set_data(f['TH'][it].data, z)
        line_THt_LES.set_data(f['TH_t'][it].data, z)
        # line_EDF_LES.set_data(data_EDF[it], z)
        line_THF_LES.set_data(Cp*RHO*f['WTH'][it].data, z)
        line_Wt_LES.set_data(f['W_t'][it].data, z)
        line_At_LES.set_data(f['ALPHA_t'][it].data, z)
        # line_Ft.set_data(data_Ft[it], z)
        # line_Et.set_data(data_Et[it], z)
        # line_Dt.set_data(data_Dt[it], z)

ani = Player(fig, update_plot, maxi=nt-1,interval=20)

# ani = animation.FuncAnimation(fig,update_plot, frames=len(les_T_t),interval=30, blit=True, repeat=True)
plt.show()