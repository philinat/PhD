#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:10:05 2024

@author: philippotn
"""
## A faire fonction extend pour remplacer les lignes du calcul
## TH_[1:-1] = (TH[1:]+TH[:-1])/2 ; TH_[0] = 1.5*TH[0]-0.5*TH[1] ; TH_[-1] = 1.5*TH[-1]-0.5*TH[-2]
## Pour contruire la liste nz+1 mais toujours avec les nan

import numpy as np
import matplotlib.pyplot as plt
from numba import njit,prange
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import matplotlib.colors as mcolors
import xarray as xr
from math import pi,cos,sin,sqrt,exp,log
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
        self.slider = matplotlib.widgets.Slider(sliderax, '',self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self,i):
        self.i = int(self.slider.val)
        self.func(self.i)
        
    def update(self,i):
        self.slider.set_val(i)

def interactive_legend(ax=None):
    if ax is None:
        ax = plt.gca()
    if ax.legend_ is None:
        ax.legend()

    return InteractiveLegend(ax.get_legend())

class InteractiveLegend(object):
    def __init__(self, legend):
        self.legend = legend
        self.fig = legend.axes.figure

        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()

        self.update()

    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legend_handles:
            artist.set_picker(10) # 10 points tolerance

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legend_handles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))

        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist

        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))

        return lookup_artist, lookup_handle

    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:

            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()

    def on_click(self, event):
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return

        for artist in self.lookup_artist.values():
            artist.set_visible(visible)
        self.update()

    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
            else:
                handle.set_visible(False)
        self.fig.canvas.draw()

    def show(self):
        plt.show()
        
from matplotlib.backend_tools import ToolBase , ToolToggleBase
class Param_tool(ToolToggleBase):
    # default_keymap = 'control'
    description = 'Turn on/off param lines'
    default_toggled = True
    def __init__(self,*args, lines, **kwargs):
        self.lines = lines
        super().__init__(*args, **kwargs)
    def enable(self,*args):
        for line in self.lines:
            line.set_visible(True)
    def disable(self, *args):
        for line in self.lines:
            line.set_visible(False)

from Module_LES import Rd,g,p0,Cp#,Rv,Lv

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
    P = p0*np.ones(n)
    integral = 0.
    for i in range(1,n):
        integral += 2*(z[i]-z[i-1])/(T[i]+T[i-1])
        P[i] *= np.exp(-g/Rd*integral)
    return P , P/Rd/T , T*(p0/P)**(Rd/Cp)

@njit()
def compute_(VAR):
    ## TH_ = np.concatenate(( [1.5*TH[0]-0.5*TH[1]] , (TH[1:]+TH[:-1])/2 , [1.5*TH[-1]-0.5*TH[-2]] ))
    nz = len(VAR)
    VAR_ = np.full(nz+1,np.nan)
    iz=0
    while np.isnan(VAR[iz]):
        iz+=1
    VAR_[iz] = VAR[iz]# 1.5*VAR[iz]-0.5*TH[iz+1] if not(np.isnan(TH[iz+1])) else VAR[iz]
    iz+=1
    while iz<nz and not(np.isnan(VAR[iz])):
        VAR_[iz] = (VAR[iz]+VAR[iz-1])/2
        iz+=1
    VAR_[iz] = VAR[iz-1] # 1.5*VAR[iz-1]-0.5*VAR[iz-2] if not(np.isnan(TH[iz-2])) else VAR[iz-1] 
    return VAR_

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
def iter_ED_TH(dt,z,z_,alpha_,alpha,dalpha,TH,RHO,RHO_,zi,H,Wstar):
    nz, = z.shape
    K = 0.4 /1 # karman constant for eddy diffusivity defined as  K*Wstar*z_*np.maximum( 0.,1-z_/zi)**2
    # pre-calculs
    K_ = K*Wstar*z_*np.maximum( 0.,1-z_/zi/1.1)**2
    # K_ = K*Wstar*np.sqrt(z_+10)*np.exp(-(z_/zi)**2)#np.maximum( 0.,1-z_/zi)**2
    # print(np.max(K_))
    arK_ =  alpha_ * RHO_ * K_
    arK_dz = arK_[1:-1] / (z[1:]-z[:-1]) # lenght nz-1
    dt_ardz = (dt/alpha/dz/RHO)[:-1] # lenght nz-1
    dt_ardz[alpha[:-1]==0] = 0.
    # preparation des diagonales a,b,c et d # avec TH[-1] fixé
    a = -dt_ardz[1:] * arK_dz[:-1]
    c = -dt_ardz * arK_dz
    b = 1-c
    b[1:] -= a
    d = TH[:-1] + dt_ardz * dalpha[:-1] * H/Cp
    d[-1] += c[-1] * TH[-1]
    TH[:-1] = TDMAsolver(a, b, c, d)
    EDFlux = -(RHO_*K_)[1:-1] * (TH[1:]-TH[:-1])/(z[1:]-z[:-1])
    return TH,EDFlux

@njit()
def iter_EDMF_TH(dt,z,z_,alpha_,alpha,dalpha,MF,MFF,TH,RHO,RHO_,zi,H,Wstar):
    nz, = z.shape
    K = 0.4 /1 # karman constant for eddy diffusivity defined as  K*Wstar*z_*np.maximum( 0.,1-z_/zi)**2
    # pre-calculs
    K_ = K*Wstar*(z_+10)*np.maximum( 0.,1-z_/zi/1.1)**2
    # K_ = K*Wstar*np.sqrt(z_+10)*np.exp(-(z_/zi)**2)#np.maximum( 0.,1-z_/zi)**2
    # print(np.max(K_))
    arK_ =  alpha_ * RHO_ * K_
    arK_dz = arK_[1:-1] / (z[1:]-z[:-1]) # lenght nz-1
    dt_ardz = (dt/alpha/dz/RHO)[:-1] # lenght nz-1
    # preparation des diagonales a,b,c et d # avec TH[-1] fixé
    a = -dt_ardz[1:] * arK_dz[:-1]
    c = -dt_ardz * arK_dz
    b = 1-c
    b[1:] -= a
    d = TH[:-1] + dt_ardz * ( dalpha[:-1] * H/Cp + alpha_[:-2]*MFF[:-2] - alpha_[1:-1]*MFF[1:-1] )
    d[-1] += c[-1] * TH[-1]
    
    TH[:-1] = TDMAsolver(a, b, c, d)
    EDFlux = -(RHO_*K_)[1:-1] * (TH[1:]-TH[:-1])/(z[1:]-z[:-1])
    return TH,EDFlux

@njit()
def diag_zi(z,dz,alpha,TH,RHO): # Calcul de zi = altitude de l'inversion = Boundary Layer Height
    nz, = z.shape
    deltaTH = 0.3 # seuil pour la détection de l'inversion
    iz = 0
    while alpha[iz]==0:
        iz+=1
    mean_TH = TH[iz]
    mass = 0.
    while iz < nz-1 and TH[iz] < mean_TH + deltaTH:
        layer_mass = alpha[iz] * RHO[iz] * dz[iz]
        mean_TH = ( mean_TH*mass + TH[iz]*layer_mass ) / ( mass+layer_mass)
        mass += layer_mass
        iz+=1
    if iz > 0:
        a = ( TH[iz] - mean_TH - deltaTH ) / (TH[iz]-TH[iz-1])
        return a*z[iz-1] + (1-a)*z[iz]
    else:
        return z[0]
@njit()
def MFs_katabatic(z,dz,TH,TH_,RHO,RHO_,zi,Wstar,H,alpha,alpha_,adot,adot_,phi,phi_,gamma):
    nz, = z.shape
    THt,Wt,THs,Ws = [ np.full(nz+1,np.nan) for i in range(4) ]
    FTH, At,Ft ,As,Fs = [ np.zeros(nz+1) for i in range(5) ]
    Et,Dt ,Es,Ds = [ np.zeros(nz) for i in range(4) ]
    # Constantes de la param
    a = 0.85 # Coefficient de réduction de flottabilité
    D = 20. # depth of surface layer (m)
    Cd = 0.005 # drag coefficient # sans dimension # wikipedia for flat plate with Re>10e6
    ############## Partie Couche de surface 
    Drag = Cd/D/np.maximum(0.001,np.sin(phi))
    iz=nz-1
    while adot[iz] == 0: 
        iz-=1
    # # Conditions initiales
    Ws[iz+1] = 0.#- sin(phi[iz]) * 0.2
    THs[iz+1] = TH[iz]-0.1
    As = adot_*D/np.cos(phi_)
    Fs[iz+1] = As[iz+1]*RHO_[iz+1]*Ws[iz+1]
    
    # # Evolution verticale
    while adot[iz] > 0 and phi[iz]>0: # boucle sur la surface
        # Préparation des 
        sinphi = sin(phi[iz])
        Ws2 = Ws[iz+1]**2
        THB = THs[iz+1] #+ dz[iz]/2 * H*cos(phi[iz])/RHO[iz]/Cp/D/Ws[iz]
        THe = TH[iz]# (TH[iz]+THs[iz+1])/2
        Bs = a * sinphi**2* g * (THB-THe)/THe # * a # réduction de flotabilité optionnelle
        
        d_rho_as = As[iz]*RHO_[iz]-As[iz+1]*RHO_[iz+1]
        # Es[iz] =  dz[iz]*Fs[iz+1]*Bs/2/Ws2 - Ws[iz+1]*max(0, d_rho_as)/2
        Ds[iz] = -dz[iz]*Fs[iz+1]*Drag[iz] - Ws[iz+1]*max(0,-d_rho_as)  
        # print(Ds[iz])
        #### Ws2 implicite
        Ws2 = (Ws2 - dz[iz]*Bs) / (1 + dz[iz]*2*Drag[iz] + max(0, As[iz]-As[iz+1])/As[iz+1] ) # Version implicite
        Ws[iz] = -sqrt(Ws2)
        Fs[iz] = As[iz]*Ws[iz]*RHO_[iz]
        Es[iz] = Fs[iz+1] - Fs[iz] + Ds[iz]
        G = 1#-exp(-5*phi[iz])
        THs[iz] = ( -Fs[iz+1]*THs[iz+1] + Es[iz]*THe + dalpha[iz]*H/Cp*G ) / (-Fs[iz]+Ds[iz]) #
        FTH[iz] += Fs[iz] * (THs[iz]-TH_[iz])
        iz-=1
    return FTH*Cp ,THt,Wt,At,Ft,Et,-Dt ,THs,Ws,As,Fs,Es,-Ds

@njit()
def MFst(z,dz,TH,TH_,RHO,RHO_,zi,Wstar,H,alpha,alpha_,adot,adot_,phi,phi_,gamma):
    nz, = z.shape
    THs,Ws = [ np.full(nz+1,np.nan) for i in range(2) ]
    FTH, THt,Wt,At,Ft ,As,Fs = [ np.zeros(nz+1) for i in range(7) ]
    Et,Dt ,Es,Ds = [ np.zeros(nz) for i in range(4) ]
    
    # Constantes de la param
    a = 0.85 # Coefficient de réduction de flottabilité
    Cw0 = 0.2
    At0 = 0.15 # EDKF At0*Cw0 = 0.065
    D0 = 0.0006
    E0 = 0.#D0
    CD = 5.
    pat = (CD/a-1)/2
    b = 5e-4#0.5/zi # (m-1)
    D = 20. # depth of surface layer (m)
    Cd = 0.005 # drag coefficient # sans dimension # wikipedia for flat plate with Re>10e6
    explicite = True
    ############## Partie Couche de surface 
    # # Dadot = 1/adot ; Dadot[0]*=adot[0]/z[0] ; Dadot[1:]*=(adot[1:]-adot[:-1])/(z[1:]-z[:-1])
    # Dadot = (adot_[1:]-adot_[:-1])/dz/adot
    # Eadot = np.maximum(0.,Dadot)/2 ; Dadot = np.maximum(0.,-Dadot)
    
    # # Dphi = np.tan(phi) ; Dphi[0]*=phi[0]/z[0] ; Dphi[1:]*=(phi[1:]-phi[:-1])/(z[1:]-z[:-1])
    # Dphi = np.tan(phi) * (phi_[1:]-phi_[:-1])/dz
    # Ephi = np.maximum(0.,Dphi)/2 ; Dphi = np.maximum(0.,-Dphi)
    
    Drho = (RHO_[:-1]-RHO_[1:])/dz/RHO
    Drag = Cd/D/np.maximum(0.001,np.sin(phi))
    
    # # Es = Eadot+Ephi
    # Ds = Drag+Drho #+Dadot+Dphi
    
    # # plt.figure()
    # # plt.plot(Dadot,z,label='$\\frac{1}{\dot\\alpha}\\frac{d\dot\\alpha}{dz}$')
    # # plt.plot(Dphi,z,label='$\\tan{\\varphi}\\frac{d\\varphi}{dz}$')
    # # plt.plot(Drho,z,label='$\\frac{1}{\\rho}\\frac{d\\rho}{dz}$')
    # # plt.plot(Drag,z,label='$\\frac{C_d}{D\sin\\phi}$')
    # # plt.legend()
    
    iz=0
    while adot[iz] == 0: 
        iz+=1
    
    ###### Partie couche de surface pour le cas plat
    # iz += 1
    # wt = ( g/THe * H/RHO_[iz]/Cp * max(0.,zi-z[iz]) )**(1/3)
    # ft = At0 * RHO_[iz]*Ws[iz]
    # FTH[iz] += ft * (THs[iz]-TH_[iz])
    
    
    ######  Partie brise pour Conditions initiales
    Ws[iz] = 0.1*max(0.001,sin(phi[iz])) * Cw0 * Wstar
    THs[iz] = TH[iz]+0.1
    As = adot_*D/np.cos(phi_)
    Fs[iz] = As[iz]*RHO_[iz]*Ws[iz]
    
    # # Evolution verticale
    # for iz in range()
    while adot[iz] > 0: # boucle sur la surface
        # Préparation des 
        sinphi = sin(phi[iz])
        Ws2 = Ws[iz]**2
        THe = TH[iz]#(TH[iz]+THs[iz])/2
        THB = THs[iz] #+ dz[iz]/2 * H*cos(phi[iz])/RHO[iz]/Cp/D/Ws[iz]
        Bs = a * sinphi**2* g * (THB-THe)/THe # * a # réduction de flotabilité optionnelle
        
        # Et[iz] = Eadot[iz]+Ephi[iz]
        # Dt[iz] = Drag[iz]+Drho[iz]+Dphi[iz]+Dadot[iz]
        
        # Fs[iz+1] = Fs[iz]*(1+dz[iz]*(Es[iz]-Ds[iz])) # Version explicite
        # Fs[iz+1] = Fs[iz]*exp(dz[iz]*(Es[iz]-Ds[iz])) # Version exponentielle
        
        Wstar =  ( g/THe * H/RHO_[iz]/Cp * max(0.,zi-z[iz]) )**(1/3)
        # Wstar =  max(D,zi-z[iz])/1000
        wt = Ws[iz] + Cw0*Wstar*dz[iz]/2/D   #### <---- initialisation thermique
        # Wt[iz] = wt
        Dsp = RHO_[iz]*dalpha[iz]*At0*wt # 
        d_rho_as = As[iz+1]*RHO_[iz+1]-As[iz]*RHO_[iz]
        # Es[iz] = dz[iz]*Fs[iz]*Bs/2/Ws2 + Ws[iz]*max(0, d_rho_as)/2 + Dsp/2
        Ds[iz] = dz[iz]*Fs[iz]*Drag[iz] + Ws[iz]*max(0,-d_rho_as)   + Dsp
        ft = Ds[iz] #### <---- initialisation thermique
        
        #### Old calcul de Ws2
        # Ws2 = (Ws2 + dz[iz]*Bs) / (1+2*dz[iz]*(Eadot[iz]+Ephi[iz]+Drag[iz])) # Version implicite
        # Ws2 += 2*dz[iz]* ( Bs -(Es[iz]+Drag[iz])*Ws2 ) # Version explicite
        #### Ws2 implicite
        Ws2 = (Ws2 + dz[iz]*Bs) / (1 + dz[iz]*2*Drag[iz] + Dsp/Fs[iz]+ max(0, As[iz+1]-As[iz])/(As[iz]+As[iz+1])*2 ) #/As[iz]) # Version implicite   
        Ws[iz+1] = sqrt(Ws2)
        Fs[iz+1] = As[iz+1]*Ws[iz+1]*RHO_[iz+1]
        Es[iz] = Fs[iz+1] - Fs[iz] + Ds[iz]
        # Ds[iz] = -Fs[iz+1] + Fs[iz] + Es[iz]
        #### Ws à partir de Es et Ds et l'équation du flux
        # Fs[iz+1] = Fs[iz] + Es[iz] - Ds[iz]
        # Ws[iz+1] = Fs[iz+1]/As[iz+1]/RHO_[iz+1]
        
        
        # G = min(1. , 0.1*dz[iz]/D * Fs[iz]*THs[iz] / (dalpha[iz]*H/Cp) )
        # G = min(1. , Fs[iz]  / ( Fs[iz]+Es[iz] ) )
        # G = 0. if alpha_[iz]==0. else 1.
        G = 1-exp(-5*phi[iz])
        # G = (sin(phi[iz]))**(0.2)
        # G = min(1.,exp(-Bs/10/sin(phi[iz])))
        # (K*Wstar*D)**2*sin(phi[iz]) / Bs/D**3
        THs[iz+1] = ( Fs[iz]*THs[iz] + Es[iz]*THe + dalpha[iz]*H/Cp*G ) / ( Fs[iz]+Es[iz] ) # Fs[iz+1]+Ds[iz] == Fs[iz]+Es[iz]
        
        # THs[iz+1] = THs[iz] + dz[iz]* ( Es[iz]*(THe-THs[iz]) + H*cos(phi[iz])/RHO[iz]/Cp/D/(Ws[iz]+Ws[iz+1])*2) # Version de base
        # THs[iz+1] = ( THs[iz]*(1-dz[iz]*Ds[iz]) + THe*dz[iz]*Es[iz]) / (1+dz[iz]*(Es[iz]-Ds[iz])) + dz[iz]*adot[iz]*H/Cp/ Fs[iz+1]# Version E D
        # THs[iz+1] = ( ( THs[iz]*(1-dz[iz]*Ds[iz]) + THe*dz[iz]*Es[iz])*Fs[iz] + dz[iz]*adot[iz]*H/Cp )/ Fs[iz+1] # Version flux
        
        FTH[iz+1] += Fs[iz+1] * (THs[iz+1]-TH_[iz+1])
        
        
        iz+=1
     
    
    
        ################# partie Thermique
        tht = THs[iz]
        wt2 = wt**2 
        at = ft/RHO_[iz]/wt
        # ft = Fs[iz] + Ds[iz-1]# dz Détrainement brise
    
        jz = iz
        while jz < nz: # boucle pour le thermique jusqu'en haut
            At[jz] += at
            Wt[jz] += at*wt
            THt[jz] += at*tht
            Ft[jz] += ft
            FTH[jz] += ft * (tht-TH_[jz])
                
            THe = TH[jz]  # Approximation de la surface négligeable
            
            # Bti = g * (tht-THe)/THe #; print("Bti = ",Bti, "z=",z_[jz])
            # if Bti>0:
            #     Dt[jz] = Drho[jz] + b + D0
            #     if explicite:### Version explicite
            #         Et[jz] = a/2*Bti/wt2
            #         wt2 += dz[jz]* ( a*Bti -2*b*wt2 )
            #         tht += dz[jz]*Et[jz]*(THe-tht)
            #     else:### Version trapèze
            #         Bt = (dz[jz]*a*Bti-wt2+sqrt((wt2-dz[jz]*a*Bti)**2+2*dz[jz]*wt2*Bti*(3*a+b))) / dz[jz]/(3*a+b) ; print("Bt = ",Bt)
            #         sumwt2 = (dz[jz]*a*Bt+wt2)/(1+dz[jz]*b) ; print("sumwt2 = ",sumwt2)
            #         Et[jz] = a*Bt/sumwt2
            #         wt2 = sumwt2 - wt2
            #         # tht -= dz[jz]*a*THe/g*Bt**2/sumwt2 # ou tht = 2*(THe/g*Bt + THe)-tht
            #         tht = 2*(THe/g*Bt + THe)-tht
            # else:
            #     Et[jz] = E0
            #     if explicite:### Version explicite
            #         Dt[jz] = Drho[jz] + b - CD*Bti/wt2
            #         wt2 += 2*dz[jz]* ( a*Bti -(b+E0)*wt2 )
            #         tht += dz[jz]*E0*(THe-tht)
            #     else:### Version trapèze
            #         Bt = 2*Bti/(2+dz[jz]*E0)
            #         sumwt2 = wt2 + (2*dz[jz]*a*Bt+wt2*(1-dz[jz]*(b+E0))) / (1+dz[jz]*(b+E0))
            #         Dt[jz] = Drho[jz] + b - 2*CD*Bt/sumwt2
            #         wt2 = sumwt2 - wt2
            #         tht = 2*(THe/g*Bt + THe)-tht
            # ft *= exp(dz[jz]*(Et[jz]-Dt[jz]))
            
            ### Version equation de at
            Bt = a* g * (tht-THe)/THe
            if Bt>0:
                at *= exp(-dz[jz]*D0)
                et = dz[jz]*ft*Bt/2/wt2
                oldwt2,oldft = wt2,ft
                wt2 = (wt2+  dz[jz]*Bt) / (1+2*dz[jz]*b)
                wt = sqrt(wt2)
                ft = at*RHO_[jz+1]*wt
                # et = (ft+oldft)/2 * (log(wt2/oldwt2)/2 + dz[jz]*b)
                tht = ( ft*tht + et*THe )  / (ft+et) #/ (ft[jz+1]+Ds[jz]) #
                
            else:
                # at *= exp(dz[jz]*(CD/a-1.)*Bt/wt2)
                wt2 = (wt2+2*dz[jz]*Bt)# / (1+2*dz[jz]*b)
                if wt2>0:
                    at *= (wt2/wt**2)**pat
                    wt = sqrt(wt2)
                    ft = at*RHO_[jz+1]*wt
                else:
                    break
            jz += 1
    
    for iz in range(nz):
        if At[iz]>0:
            Wt[iz] /= At[iz]
            THt[iz] /= At[iz]
        else:
            Wt[iz] = np.nan
            THt[iz] = np.nan
    return FTH ,THt,Wt,At,Ft,Et,-Dt ,THs,Ws,As,Fs,Es,-Ds

@njit()
def get_dalpha_phi_gamma(ZS,z_,dx,cyclic=True):
    # we must have : max(z_)>max(ZS)
    ny,nx = ZS.shape
    ZS_ = np.zeros((ny+2,nx+2))
    ZS_[1:-1,1:-1] = np.copy(ZS)
    ZS_[0,:] = ZS_[-2,:] if cyclic else ZS_[1,:]
    ZS_[-1,:] = ZS_[1,:] if cyclic else ZS_[-2,:]
    ZS_[:,0] = ZS_[:,-2] if cyclic else ZS_[:,1]
    ZS_[:,-1] = ZS_[:,1] if cyclic else ZS_[:,-2]
    grad_ = np.sqrt( ((ZS_[:-1,:-1]+ZS_[1:,:-1]-ZS_[:-1,1:]-ZS_[1:,1:])/2/dx)**2 + ((ZS_[:-1,:-1]+ZS_[:-1,1:]-ZS_[1:,:-1]-ZS_[1:,1:])/2/dx)**2 )
    grad = (grad_[:-1,:-1]+grad_[1:,:-1]+grad_[:-1,1:]+grad_[1:,1:])/4
    lapla = (ZS_[1:-1,:-2]+ZS_[1:-1,2:] - 2*ZS_[1:-1,1:-1])/dx**2 + (ZS_[:-2,1:-1]+ZS_[2:,1:-1] - 2*ZS_[1:-1,1:-1])/dx**2
    lapla2 = lapla**2
    
    nz = len(z_)-1
    count = np.zeros(nz,dtype=np.int64)
    sum_grad = np.zeros(nz)
    sum_lapla = np.zeros(nz)
    sum_lapla2 = np.zeros(nz)
    for iy in range(ny):
        for ix in range(nx):
            iz = 0
            while ZS[iy,ix]>z_[iz+1]:
                iz+=1
            count[iz] += 1
            sum_grad[iz] += grad[iy,ix]
            sum_lapla[iz] += lapla[iy,ix]
            sum_lapla2[iz] += lapla2[iy,ix]
    return count/(nx*ny) , sum_grad/count , np.sqrt( sum_lapla2/count - (sum_lapla/count)**2 )
    
# @njit()
# def iter_energy_budget_alpha_explicit(dt,flux_,TH,RHO,z_,dz,alpha_,H):
#     nz, = TH.shape
#     af_ = alpha_*flux_
#     dalpha = alpha_[1:]-alpha_[:-1]
#     return TH + dt*2/(alpha_[1:]+alpha_[:-1])/dz/RHO * ( af_[:-1] - af_[1:] + dalpha*H/Cp )
    
#%%  
ED = True
MF = True
LES = True

if LES:
    # simu = 'E620M' ; seg = '200m' ; zmax = 6000.
    simu = 'DMOPL' ; seg = 'M100m' ; zmax = 4000.
    userPath = '/home/philippotn'
    if simu[0] in ['A','C']: dataPath = userPath+'/Documents/SIMU_LES/'
    elif simu[0] in ['D']: dataPath = userPath+'/Documents/NO_SAVE/'
    elif simu[0] in ['E']: dataPath = userPath+'/Documents/Canicule_Août_2023/'+simu+'/'
    savePath = userPath+'/Images/LES_'+simu+'/Plot_param_vs_LES/'
    f = xr.open_dataset(dataPath+simu+'_'+seg+'_mean.nc')
    RHO , TH = f['RHO'][0].data, f['TH'][0].data ; T0=np.nanmin(TH) ; RHO0 = np.nanmax(RHO)
    # TH[np.isnan(TH)] = T0
    # RHO[np.isnan(RHO)] = RHO0
    TH_ = compute_(TH)
    RHO_ = compute_(RHO)
    # TH_ = np.concatenate(( [1.5*TH[0]-0.5*TH[1]] , (TH[1:]+TH[:-1])/2 , [1.5*TH[-1]-0.5*TH[-2]] ))
    # RHO_ = np.concatenate(( [1.5*RHO[0]-0.5*RHO[1]] , (RHO[1:]+RHO[:-1])/2 , [1.5*RHO[-1]-0.5*RHO[-2]] ))
    H = np.append( f['H'].data[:] ,0. )
    nt = len(f['time'])
    time = f['time']
    time_text = [ '{:02d}h{:02d}'.format(int(time[it].dt.hour),int(time[it].dt.minute)) for it in range(nt) ]
    time_integration = False
    dt = 600. if time_integration else 0.
    niter = int( (f['time'].data[1]-f['time'].data[0])/ np.timedelta64(1, 's') /dt ) if time_integration else 1
    z = f['z'].data ; nz = len(z)
    dz = f['dz'].data
    z_ = np.concatenate((np.zeros(1),np.cumsum(dz)))
    dalpha,phi,gamma = get_dalpha_phi_gamma(f['ZS'].data ,z_,f['x'].data[1]-f['x'].data[0])
    alpha_ = np.concatenate((np.zeros(1),np.cumsum(dalpha)))
    alpha = (alpha_[1:]+alpha_[:-1])/2
    # phi[np.isnan(phi)] = 0.
    thermal_groups = []
    # for var in f.data_vars:
    #     if var[:4] == 'W_t_' : thermal_groups.append(int(var[4:]))
    cmap_thermal_groups = matplotlib.colormaps.get_cmap('autumn_r')
    color_thermal_groups = cmap_thermal_groups(np.linspace(0,1,len(thermal_groups)))
    
    adot = dalpha/dz # alpha dot
    adot_ = compute_(adot)
    phi_ = compute_(phi)
    # adot_ = np.concatenate(( [1.5*adot[0]-0.5*adot[1]] , (adot[1:]+adot[:-1])/2 , [1.5*adot[-1]-0.5*adot[-2]] ))
    # phi_ = np.concatenate(( [1.5*phi[0]-0.5*phi[1]] , (phi[1:]+phi[:-1])/2 , [1.5*phi[-1]-0.5*phi[-2]] ))
    
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
data_TH_ = np.zeros((nt,nz+1))
data_EDF = np.zeros((nt,nz+1))
data_MFF = np.zeros((nt,nz+1))
data_THt = np.zeros((nt,nz+1))
data_THs = np.zeros((nt,nz+1))
data_Wt = np.zeros((nt,nz+1))
data_Ws = np.zeros((nt,nz+1))
data_At = np.zeros((nt,nz+1))
data_As = np.zeros((nt,nz+1))
data_Ft = np.zeros((nt,nz+1))
data_Fs = np.zeros((nt,nz+1))
data_Et = np.zeros((nt,nz))
data_Es = np.zeros((nt,nz))
data_Dt = np.zeros((nt,nz))
data_Ds = np.zeros((nt,nz))

#%%
zi = np.zeros(nt)
Wstar = np.zeros(nt)

## Boucle temporelle##
for it in range(nt):
# for it in [5]:
    if not(time_integration): TH = f['TH'][it].data
    for ct in range(niter):
        # print(TH[:10])
        # if np.any(np.isnan(TH)):
        #     print('!!! nan in TH !!!')
        #     break
        
        # flux_ = np.zeros(nz+1) # flux_ = np.zeros_like(z_)
        # K = 1e-1
        # flux_[1:-1] = - RHO_[1:-1] * K * (TH[1:]-TH[:-1])/((z[1:]-z[:-1]))
        # TH = iter_energy_budget_alpha_explicit(dt,flux_,TH,RHO,z_,dz,alpha_, H_iter )
        
        H_iter = ( (niter-ct-0.5)*H[it-1] + (ct+0.5)*H[it] )/niter if time_integration else H[it]
        zi_iter = diag_zi(z,dz,alpha,TH,RHO)
        Wstar_iter = ( g/T0 * H_iter/RHO0/Cp * zi_iter )**(1/3) if H_iter>0 else 0.1
        
        if H_iter>0 and time_integration:
            MFF,_,_,_,MF,_,_,_,_,_,_,_,_ = MFst(z,dz,TH,TH_,RHO,RHO_,zi_iter,Wstar_iter,H_iter,alpha,alpha_,adot,adot_,phi,phi_,gamma)
        if H_iter<0 and time_integration:
            MFF,_,_,_,MF,_,_,_,_,_,_,_,_ = MFs_katabatic(z,dz,TH,TH_,RHO,RHO_,zi_iter,Wstar_iter,H_iter,alpha,alpha_,adot,adot_,phi,phi_,gamma)
        
        TH_ = compute_(TH)
        # TH_[1:-1] = (TH[1:]+TH[:-1])/2 ; TH_[0] = 1.5*TH[0]-0.5*TH[1] ; TH_[-1] = 1.5*TH[-1]-0.5*TH[-2]
        # if time_integration:
        TH,EDF = iter_ED_TH(dt,z,z_,alpha_,alpha,dalpha,TH,RHO,RHO_,zi_iter,H_iter,Wstar_iter)
        # TH,EDF = iter_EDMF_TH(dt,z,z_,alpha_,alpha,dalpha,MF,MFF,TH,RHO,RHO_,zi_iter,H_iter,Wstar_iter)
        # energy += dt*H_iter
    
    zi[it] = zi_iter
    Wstar[it] = Wstar_iter
    data_EDF[it,1:-1] = EDF ; data_EDF[it,0] = H_iter
    data_TH[it] = TH
    data_TH_[it] = TH_
    
    if H_iter>0:
        data_MFF[it] ,data_THt[it],data_Wt[it],data_At[it],data_Ft[it],data_Et[it],data_Dt[it] ,data_THs[it],data_Ws[it],data_As[it],data_Fs[it],data_Es[it],data_Ds[it] = MFst(z,dz,TH,TH_,RHO,RHO_,zi_iter,Wstar_iter,H_iter,alpha,alpha_,adot,adot_,phi,phi_,gamma)
    elif H_iter<0:
        data_MFF[it] ,data_THt[it],data_Wt[it],data_At[it],data_Ft[it],data_Et[it],data_Dt[it] ,data_THs[it],data_Ws[it],data_As[it],data_Fs[it],data_Es[it],data_Ds[it] = MFs_katabatic(z,dz,TH,TH_,RHO,RHO_,zi_iter,Wstar_iter,H_iter,alpha,alpha_,adot,adot_,phi,phi_,gamma)


#%%
# import matplotlib.animation as animation
ncols = 5
ft=15

fig, axs = plt.subplots(ncols=ncols,figsize=(20,6))
plt.subplots_adjust(left=0.045,right=0.995,bottom=0.1,top=0.9,wspace=0.05,hspace=0.)
# fig.figsize = (16, 5)
colors = ['r','g','b','c'] # t,s, all
xlims = [[T0-2,T0+16],[-100,400],[-2,5],[-0.01,0.16],[-0.015,0.15],[-0.02,0.02]]
xlabels = ['$\\theta$ (K)', '$c_p \\rho \\overline{w^{\prime} \\theta^{\prime}}$ (W/m²)', 'W (m/s)', '$\\alpha$' ,'f','$\\epsilon$ and $\\delta$ ($m^{-1}$)']

linestyle_LES = '-.'

for i in range(ncols):
    axs[i].set_ylim([0,zmax])
    axs[i].grid(axis='y')
    axs[i].axvline(x=0,color='grey',linestyle='-',linewidth = 0.5)
    if i==0:
        axs[i].set_ylabel('z (m)',fontsize=ft)
    # else:
        # axs[i].set_yticks([])#,fontsize=0)
        # axs[i].set_yticklabels([])
    
    axs[i].set_xlabel(xlabels[i],fontsize=ft)
    axs[i].set_xlim(xlims[i])

# axs[0].axvline(x=T0,color='grey',linestyle='-',linewidth=1,label='$\\theta_{ini}$')
line_T0 = axs[0].plot(f['TH'][0].data,z,color='grey',linestyle='-',linewidth=1,label='$\overline{\\theta_{ini}}$')
line_zi = axs[0].axhline(y=zi[0],color='k',linestyle='--',label='$z_i$')
line_H = axs[1].axvline(x=H[0],color='k',linestyle='--',label='$H$')
line_Wstar = axs[2].axvline(x=Wstar[0],color='k',linestyle='--',label='$w_*$')
line_A, = axs[3].plot(alpha,z,color='k',linestyle='--',label='$\\alpha$')

line_TH, = axs[0].plot([],[],color=colors[2],label='$\overline{\\theta}$')
line_THt,= axs[0].plot([],[],color=colors[0],label='$\\theta_t$')
line_THs,= axs[0].plot([],[],color=colors[1],label='$\\theta_s$')

line_THF,= axs[1].plot([],[],color=colors[2],label='$\overline{w^{\prime}\\theta^{\prime}}_{EDMF}$')
line_THFt,= axs[1].plot([],[],color=colors[0],label='$\\beta_t \overline{w}^t (\overline{\\theta}^t - \overline{\\theta})$')
line_THFs,= axs[1].plot([],[],color=colors[1],label='$\\beta_s \overline{w}^s (\overline{\\theta}^s - \overline{\\theta})$')
line_EDF,= axs[1].plot([],[],color=colors[3],label='$\overline{w^{\prime}\\theta^{\prime}}_{ED}$') 
line_Wt, = axs[2].plot([],[],color=colors[0],label='$w_t$') 
line_Ws, = axs[2].plot([],[],color=colors[1],label='$w_s$') 
line_At, = axs[3].plot([],[],color=colors[0],label='$\\alpha_t$')
line_As, = axs[3].plot([],[],color=colors[1],label='$\\alpha_s$')
line_Ft, = axs[4].plot([],[],color=colors[0],label='$f_t$')
line_Fs, = axs[4].plot([],[],color=colors[1],label='$f_s$') 
line_Et, = axs[4].plot([],[],color=colors[0],label='$\\epsilon_t$',linestyle=':')
line_Dt, = axs[4].plot([],[],color=colors[0],label='$\\delta_t$',linestyle=':')
line_Es, = axs[4].plot([],[],color=colors[1],label='$\\epsilon_s$',linestyle=':')
line_Ds, = axs[4].plot([],[],color=colors[1],label='$\\delta_s$',linestyle=':')

param_lines = [line_TH,line_THt,line_THs,line_THF,line_Wt,line_Ws,line_At,line_As]

if LES:
    # all atmosphere profiles
    line_TH_LES, = axs[0].plot([],[],color=colors[2],label='$\overline{\\theta}^{LES}$',linestyle=linestyle_LES)
    line_THF_LES,= axs[1].plot([],[],color=colors[2],label='$\overline{w^{\prime}\\theta^{\prime}}^{LES}$' ,linestyle=linestyle_LES)
    # thermals profiles
    line_THt_LES,= axs[0].plot([],[],color=colors[0],label='$\\theta_t^{LES}$',linestyle=linestyle_LES)
    line_THFt_LES,= axs[1].plot([],[],color=colors[0],label='$\\beta_t \overline{w}^t (\overline{\\theta}^t - \overline{\\theta})^{LES}$' ,linestyle=linestyle_LES)
    line_Wt_LES, = axs[2].plot([],[],color=colors[0],label='$w_t^{LES}$',linestyle=linestyle_LES) 
    line_At_LES, = axs[3].plot([],[],color=colors[0],label='$\\alpha_t^{LES}$',linestyle=linestyle_LES)
    line_Ft_LES, = axs[4].plot([],[],color=colors[0],label='$f_t^{LES}$',linestyle=linestyle_LES)
    if len(thermal_groups)>0:
        line_THt_groups_LES,line_THFt_groups_LES,line_Wt_groups_LES,line_At_groups_LES = [],[],[],[]
        for ig,group in enumerate(thermal_groups):
            line_THt_groups_LES.append(axs[0].plot([],[],color=color_thermal_groups[ig],linestyle=linestyle_LES)[0]) # ,label='$\\theta_{t_{'+str(g)+'}}^{LES}$'
            line_THFt_groups_LES.append(axs[1].plot([],[],color=color_thermal_groups[ig],linestyle=linestyle_LES)[0]) # 
            line_Wt_groups_LES.append(axs[2].plot([],[],color=color_thermal_groups[ig],linestyle=linestyle_LES)[0]) # ,label='$w_{t_{'+str(g)+'}}^{LES}$'
            line_At_groups_LES.append(axs[3].plot([],[],color=color_thermal_groups[ig],linestyle=linestyle_LES)[0]) # ,label='$\\alpha_{t_{'+str(g)+'}}^{LES}$'
    
    # surface layer profiles
    line_THs_LES,= axs[0].plot([],[],color=colors[1],label='$\\theta_s^{LES}$',linestyle=linestyle_LES)
    line_THFs_LES,= axs[1].plot([],[],color=colors[1],label='$\\beta_s \overline{w}^s (\overline{\\theta}^s - \overline{\\theta})^{LES}$' ,linestyle=linestyle_LES)
    line_Ws_LES, = axs[2].plot([],[],color=colors[1],label='$w_s^{LES}$',linestyle=linestyle_LES) 
    line_As_LES, = axs[3].plot([],[],color=colors[1],label='$\\alpha_s^{LES}$',linestyle=linestyle_LES)
    line_Fs_LES, = axs[4].plot([],[],color=colors[1],label='$f_s^{LES}$',linestyle=linestyle_LES)
    LES_lines = [line_TH_LES,line_THt_LES,line_THs_LES,line_THF_LES,line_Wt_LES,line_Ws_LES,line_At_LES,line_As_LES]

legends = []  
for ax in axs:
    # ax.legend()
    legends.append(interactive_legend(ax))
    
    

title = fig.suptitle(simu+'  '+time_text[0])
alpha_1 = np.copy(alpha_) ; alpha_1[0] = 1.
def update_plot(it):
    title.set_text(simu+'  '+time_text[it])
    line_zi.set_data([0,1],[zi[it],zi[it]])
    line_H.set_data([H[it],H[it]],[0,1])
    line_Wstar.set_data([Wstar[it],Wstar[it]],[0,1])
    
    line_TH.set_data(data_TH[it], z)
    # line_TH.set_data(data_TH_[it], z_)
    line_THt.set_data(data_THt[it], z_)
    line_THs.set_data(data_THs[it], z_)
    line_EDF.set_data(Cp*data_EDF[it], z_)
    line_THF.set_data(Cp*(data_EDF[it]+data_MFF[it]), z_)
    line_THFt.set_data(Cp*RHO_*data_At[it].data/alpha_1*data_Wt[it]*(data_THt[it]-data_TH_[it]), z_)
    line_THFs.set_data(Cp*RHO_*data_As[it].data/alpha_1*data_Ws[it]*(data_THs[it]-data_TH_[it]), z_)
    line_Wt.set_data(data_Wt[it], z_)
    line_Ws.set_data(data_Ws[it], z_)
    line_At.set_data(data_At[it], z_)
    line_As.set_data(data_As[it], z_)
    line_Ft.set_data(data_Ft[it], z_)
    line_Fs.set_data(data_Fs[it], z_)
    line_Et.set_data(data_Et[it], z)
    line_Dt.set_data(data_Dt[it], z)
    line_Es.set_data(data_Es[it], z)
    line_Ds.set_data(data_Ds[it], z)
    if LES:
        line_TH_LES.set_data(f['TH'][it].data, z)
        line_THt_LES.set_data(f['TH_t'][it].data, z)
        line_THs_LES.set_data(f['TH_s'][it].data, z)
        # line_EDF_LES.set_data(data_EDF[it], z)
        line_THF_LES.set_data(Cp*RHO*f['WTH'][it].data, z)
        line_THFt_LES.set_data(Cp*RHO*f['ALPHA_t'][it].data/f['ALPHA'][it].data*f['W_t'][it].data*(f['TH_t'][it].data-f['TH'][it].data), z)
        line_THFs_LES.set_data(Cp*RHO*f['ALPHA_s'][it].data/f['ALPHA'][it].data*f['W_s'][it].data*(f['TH_s'][it].data-f['TH'][it].data), z)
        line_Wt_LES.set_data(f['W_t'][it].data, z)
        line_Ws_LES.set_data(f['W_s'][it].data, z)
        line_At_LES.set_data(f['ALPHA_t'][it].data, z)
        line_As_LES.set_data(f['ALPHA_s'][it].data, z)
        # line_TKEt_LES.set_data(f['TKE_t'][it].data, z)
        # line_TKEs_LES.set_data(f['TKE_s'][it].data, z)
        line_Ft_LES.set_data(RHO*f['ALPHA_t'][it].data*f['W_t'][it].data, z)
        line_Fs_LES.set_data(RHO*f['ALPHA_s'][it].data*f['W_s'][it].data, z)
        # line_Et.set_data(data_Et[it], z)
        # line_Dt.set_data(data_Dt[it], z)
        
        for ig,group in enumerate(thermal_groups):
            line_THt_groups_LES[ig].set_data(f['TH_t_'+str(group)][it].data, z)
            line_THFt_groups_LES[ig].set_data(Cp*RHO*f['ALPHA_t_'+str(group)][it].data/f['ALPHA'][it].data*f['W_t_'+str(group)][it].data*(f['TH_t_'+str(group)][it].data-f['TH'][it].data), z)
            line_Wt_groups_LES[ig].set_data(f['W_t_'+str(group)][it].data, z)
            line_At_groups_LES[ig].set_data(f['ALPHA_t_'+str(group)][it].data, z)

ani = Player(fig, update_plot, maxi=nt-1,interval=20)

# ani.fig.canvas.manager.toolmanager.add_tool('Param',Param_tool,lines=param_lines)
# ani.fig.canvas.manager.toolbar.add_tool('Param', 'animation')

# ani = animation.FuncAnimation(fig,update_plot, frames=len(les_T_t),interval=30, blit=True, repeat=True)
plt.show()

#%%
# capacity = (Cp*alpha*dz*RHO).reshape(1,nz)

# plt.figure()
# plt.plot(time,np.sum(capacity *data_TH ,axis=1) ,label="param" )
# plt.plot(time,np.sum(capacity *f['TH'].data ,axis=1) ,label="LES" )
# plt.plot(time,f['TSE'].data  ,label="TSE" )
# # plt.plot(time,f['TSE'].data[0] + np.cumsum(H[:-1])*dt*niter  ,label="integral H" )
# plt.legend()
# hours = matplotlib.dates.HourLocator()
# ax = plt.gca()
# ax.xaxis.set_major_locator(hours)
# # ax.xaxis.set_minor_locator(days)
# ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(hours,formats=['%Y', '%b', '%b/%d', '%H', '%H:%M', '%S.%f']))
# # ax.xaxis.set_minor_formatter(matplotlib.dates.ConciseDateFormatter(days))
# plt.xticks(fontsize=ft)

#%% Test méthode numérique du thermique
# a = 0.85 # Coefficient de réduction de flottabilité
# Cw0 = 0.2
# At0 = 0.15 # EDKF At0*Cw0 = 0.065
# D0 = 0.0005
# E0 = 0.#D0
# CD = 5.
# pat = (CD/a-1)/2
# b = 5e-4#0.5/zi # (m-1)
# D = 20. # depth of surface layer (m)
# Cd = 0.005 # drag coefficient # sans dimension # wikipedia for flat plate with Re>10e6

# def iter_t_explicite(dz,THe,wt2,tht,ft):
#     Bti = g * (tht-THe)/THe #; print("Bti = ",Bti, "z=",z_[jz])
#     if Bti>0:
#         Dt = b + D0
#         Et = a/2*Bti/wt2
#         wt2 += dz* ( a*Bti -2*b*wt2 )
#         tht += dz*Et*(THe-tht)
#     else:
#         Et = E0
#         Dt = b - CD*Bti/wt2
#         wt2 += 2*dz* ( a*Bti -(b+E0)*wt2 )
#         tht += dz*E0*(THe-tht)
#     ft *= exp(dz*(Et-Dt))
#     return wt2,tht,ft

# def iter_t_trapeze(dz,THe,wt2,tht,ft):
#     Bti = g * (tht-THe)/THe #; print("Bti = ",Bti, "z=",z_)
#     if Bti>0:
#         Dt = b + D0
#         Bt = (dz*a*Bti-wt2+sqrt((wt2-dz*a*Bti)**2+2*dz*wt2*Bti*(3*a+b))) / dz/(3*a+b) #; print("Bt = ",Bt)
#         sumwt2 = (dz*a*Bt+wt2)/(1+dz*b) #; print("sumwt2 = ",sumwt2)
#         Et = a*Bt/sumwt2
#         wt2 = sumwt2 - wt2
#         # tht -= dz*a*THe/g*Bt**2/sumwt2 # ou tht = 2*(THe/g*Bt + THe)-tht
#         tht = 2*(THe/g*Bt + THe)-tht
#     else:
#         Et = E0
#         Bt = 2*Bti/(2+dz*E0)
#         sumwt2 = wt2 + (2*dz*a*Bt+wt2*(1-dz*(b+E0))) / (1+dz*(b+E0))
#         Dt = b - 2*CD*Bt/sumwt2
#         wt2 = sumwt2 - wt2
#         tht = 2*(THe/g*Bt + THe)-tht
#     ft *= exp(dz*(Et-Dt))
#     return wt2,tht,ft

# def iter_t_rat(dz,THe,wt2,tht,ft):
#     Bt = a* g * (tht-THe)/THe
#     rat = ft/sqrt(wt2)
#     oldwt2,oldft = wt2,ft
#     if Bt>0:
#         rat *= exp(-dz*D0)
#         et = dz*ft*Bt/2/wt2
#         wt2 = (wt2+  dz*Bt) / (1+2*dz*b)
#         wt = sqrt(wt2)
#         ft = rat*wt
#         # et = (ft+oldft)/2 * (log(wt2/oldwt2)/2 + dz*b)
#         tht = ( ft*tht + et*THe )  / (ft+et) #/ (ft[jz+1]+Ds) #
        
#     else:
#         # at *= exp(dz*(CD/a-1.)*Bt/wt2)
#         wt2 = (wt2+2*dz*Bt)#/ (1+2*dz*b)
#         if wt2>0:
#             rat *= (wt2/oldwt2)**pat
#             wt = sqrt(wt2)
#             ft = rat*wt
#         else:
#             ft=0
#     return wt2,tht,ft


# plt.close('all')
# it = 7
# TH = f['TH'][it].data
# TH_ = compute_(TH)

# ncols = 5
# ft=15

# fig, axs = plt.subplots(ncols=ncols,figsize=(20,6))
# plt.subplots_adjust(left=0.045,right=0.995,bottom=0.1,top=0.9,wspace=0.05,hspace=0.)
# # fig.figsize = (16, 5)
# # colors = ['r','g','b','c'] # t,s, all
# xlims = [[T0-2,T0+16],[-100,400],[-2,5],[-0.01,0.21],[-0.5,0.5],[-0.02,0.02]]
# for i in range(len(axs)):
#     axs[i].set_xlim(xlims[i])
#     axs[i].set_ylim([0,zmax])
#     axs[i].grid(axis='y')
#     axs[i].axvline(x=0,color='grey',linestyle='-',linewidth = 0.5)
#     if i==0:
#         axs[i].set_ylabel('z (m)',fontsize=ft)
#     axs[i].set_xlabel(xlabels[i],fontsize=ft)
#     axs[i].set_xlim(xlims[i])
    

# axs[0].plot(TH,z,'--k')
# axs[0].plot(f['TH_t'][it].data,z,'-k')
# axs[1].plot(Cp*f['WTH'][it].data,z,'--k')
# axs[1].plot(Cp*RHO*f['ALPHA_t'][it].data/f['ALPHA'][it].data*f['W_t'][it].data*(f['TH_t'][it].data-f['TH'][it].data),z,'-k')
# axs[2].plot(f['W_t'][it].data,z,'-k')
# axs[3].plot(f['ALPHA_t'][it].data,z,'-k')
# axs[4].plot(RHO*f['W_t'][it].data*f['ALPHA_t'][it].data,z,'-k')


# # axs[0].plot(TH,z,'-k')
# for iter_t in [iter_t_explicite,iter_t_rat]:
#     THt = np.full(nz+1,np.nan)
#     Wt = np.full(nz+1,np.nan)
#     Ft = np.zeros(nz+1)
#     At = np.zeros(nz+1)
#     FTHt = np.zeros(nz+1)
#     iz = 1
    
#     wt = 0.2
#     at = 0.15
#     ft = RHO_[iz]*at*wt
#     wt2 = wt**2
#     tht = TH[iz-1]
    
#     jz = iz
#     while jz < nz and wt2>0: # boucle pour le thermique jusqu'en haut
#         THt[jz] = tht
#         FTHt[jz] = ft * (tht-TH_[jz])
#         Wt[jz] = sqrt(wt2)
#         At[jz] = ft/Wt[jz]/RHO[jz]
#         Ft[jz] = ft
        
        
#         wt2,tht,ft = iter_t(dz[jz],TH[jz],wt2,tht,ft)
#         jz += 1
    
#     axs[0].plot(THt,z_)
#     axs[1].plot(FTHt*Cp,z_)
#     axs[2].plot(Wt,z_)
#     axs[3].plot(At,z_)
#     axs[4].plot(Ft,z_)
