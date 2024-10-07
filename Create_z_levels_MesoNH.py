#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:10:21 2023

@author: philippotn
"""
import matplotlib.pyplot as plt
import numpy as np

dz0 = 4

dx = 600
ZTOP = 20_000

dzBL = 200
ZBL = 5000
# ZBL = ZTOP

# ZEPONGE = 16_000
# ZEPONGE = ZBL
ZEPONGE = ZTOP


stretching = 1.1 # increase of dz by 20% 

z = 0.
dz = dz0
list_z = [z]
while z < ZTOP:
    z += dz
    list_z.append(z)
    if z<ZBL:
        if dz<dzBL:
            dz = min(dzBL,dz*stretching)
    elif dz<dx:
        dz = min(dx,dz*stretching)
    if z>ZEPONGE:
        dz *= stretching
    
list_z = np.array(list_z)
nz = len(list_z)
list_dz = list_z[1:]-list_z[:-1]
plt.plot(np.arange(nz),list_z )
plt.plot(list_dz,list_z[1:])

print('NKMAX='+str(nz-1))
#%%
#list_z = np.linspace(0,20000,201)
savePath = '/home/philippotn/Documents/PhD/ZHAT_txt/'
with open(savePath+'ZHAT'+'_'+str(dz0)+'_'+str(round((stretching-1)*100))+'%_'+str(dzBL)+'_'+str(ZBL)+'_'+str(dx)+'_'+str(ZTOP)+'.txt','w') as f:
    f.write('NKMAX='+str(nz-1)+'\n')
    f.write('ZHAT\n')
    for z in list_z:
        f.write("{:.3f}\n".format(z))
        print("{:.3f}".format(z),end=" ")
