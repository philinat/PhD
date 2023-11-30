#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:10:21 2023

@author: philippotn
"""
import matplotlib.pyplot as plt
import numpy as np

dx = 200

dzBL = 50
ZBL = 3500

ZTOP = 20_000

# ZEPONGE = 16_500
ZEPONGE = ZTOP

z = 0.
dz = 10.
stretching = 1.15 # increase of dz by 20% 

list_z = np.array([z])
while z < ZTOP:
    z += dz
    if z<ZBL:
        if dz<dzBL:
            dz = min(dzBL,dz*stretching)
    elif dz<dx:
        dz = min(dx,dz*stretching)
    if z>ZEPONGE:
        dz *= stretching
    list_z = np.append(list_z,z)

nz = len(list_z)
list_dz = list_z[1:]-list_z[:-1]
plt.plot(np.arange(nz),list_z )
plt.plot(list_dz,list_z[1:])

#%%
#list_z = np.linspace(0,20000,201)
print('NKMAX='+str(nz-1))
with open('ZHAT'+str(dx)+'.txt','w') as f:
    f.write('NKMAX='+str(nz-1)+'\n')
    f.write('ZHAT\n')
    for z in list_z:
        f.write("{:.1f}\n".format(z))
        print("{:.1f}".format(z),end=" ")
