#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Aug 2022

@author: Annabella MEECH & Baptiste KLEIN
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pickle
from functions import *

outroot = "Input_data/igrins/"
mode_pca    = "pca"
plot_all_orders = True

if mode_pca == "pca" or mode_pca == "PCA":
    outroot  += "PCA/"
    filename  = outroot+"reduced_1.pkl"
    file_info = outroot+"info_1.dat"
    if not os.path.exists(outroot):
        sys.exit("No file found, run reduce_data.py first")

with open(filename,'rb') as specfile:
    A = pickle.load(specfile)
orders,WW,Ir,T_obs,phase,window,berv,vstar,airmass,SN = A

fac = 2.0 # factor of std at which to mask
outroot += 'masked/'
if not os.path.exists(outroot):
    os.makedirs(outroot)


### Create order objects
nord     = len(orders)
print(nord,"orders detected")
list_ord = []
for nn in range(nord):
    O        = Order(orders[nn])
    O.W_red  = np.array(WW[nn],dtype=float)
    O.I_red  = np.array(Ir[nn],dtype=float)
    #O.blaze  = np.array(blaze[nn],dtype=float)
    #O.I_atm  = np.array(Ia[nn],dtype=float)
    O.SNR    = np.array(SN[nn],dtype=float)
    O.W_mean = O.W_red.mean()
    list_ord.append(O)
print("DONE\n")

### Mask strong residual telluric lines
for nn in range(nord):
    if plot_all_orders:
        plot=True
    else:
        if nn==plot_ord:
            plot=True
        else:
            plot=False
    O         = list_ord[nn]
    print("ORDER",O.number)

    W_cl,I_cl = np.copy(O.W_red), np.copy(O.I_red)-1.0 #why is it centred at 1 after PCA?
    nep,npix = I_cl.shape

    if plot:
        fig,axes = plt.subplots(2,1,figsize=(8,4))
        wmin,wmax = W_cl.min(),W_cl.max()
        dw = np.average(W_cl[1:]-W_cl[:-1])
        extent = (wmin - 0.5 * dw, wmax - 0.5 * dw, nep - 0.5, 0.5)
        xlabel = 'wavelength (nm)'
        mp1=axes[0].imshow(I_cl, extent=extent, interpolation='nearest', aspect='auto')
        fig.colorbar(mp1,ax=axes[0])
        axes[0].set_title('Order {}'.format(O.number))

    # identify residual tellurics
    std = np.std(I_cl,axis=0)
    thr = fac*np.nanmedian(std)
    l   = std>thr
    l   = np.tile(l[None,:],(nep,1))
    I_cl[l] = np.nan
    O.I_mask = I_cl

    if plot:
        mp2=axes[1].imshow(O.I_mask, extent=extent, interpolation='nearest', aspect='auto')
        fig.colorbar(mp2,ax=axes[1])
        axes[1].set_xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(outroot+"masked_order{}.png".format(O.number))

print("DONE")
