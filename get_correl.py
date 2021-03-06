#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Mar 2022
Edited in Jun 2022
@authors: Baptiste KLEIN, Florian DEBRAS & Annabella MEECH
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import time
from functions import *
from correlation_fcts import *
from scipy import ndimage
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", type=str, required=True,\
        help="select which data to analyse: IGRINS or SPIROU")
    parser.add_argument("--red-mode", type=str, default="PCA",\
        help="choose reduction type, options: PCA")
    parser.add_argument("--aligned", action="store_true", default=False,\
        help="use the aligned, wavelength recalibrated spectral matrix, \
         (for IGRINS data)")
    parser.add_argument("--inject", action="store_true", default=False,\
        help="if true, use spectra with injected signal")
    parser.add_argument("--inj-Kp", type=float, default=0.0, \
        help="Kp velocity (km/s) at which to inject planet signal")
    parser.add_argument("--inj-vsys", type=float, default=0.0, \
        help="systemic velocity (km/s) at which to inject planet signal")
    parser.add_argument("--inj-amp", type=float, default=0.0, \
        help="scale factor of injected planet signal")


    args = parser.parse_args()

    model_dir = 'pRT_models/'
    data_dir  = 'Input_data/'
    if args.instrument == 'igrins' or args.instrument == 'IGRINS':
        data_dir += 'igrins/'
    elif args.instrument == 'spirou' or args.instrument == 'SPIROU':
        data_dir += 'spirou/'

    if args.inject:
        print("loading in spectra with injected signal...")
        data_dir += "inject_amp{:.1f}_Kp{:.1f}_vsys{:.1f}/".format(args.inj_amp,\
        args.inj_Kp,args.inj_vsys)
    if args.red_mode=='pca' or args.red_mode=='PCA':
        data_dir+= 'PCA/'

    if args.aligned:
        filename = "reduced_aligned.pkl"
    else:
        filename = "reduced_1.pkl"

    # model files
    species     = ['CO'] # edit to include species in model
    sp          = '_'.join(i for i in species)
    solar       = '1x'
    model_dir   = 'pRT_models/'
    model_dir  += 'aumicb_{}Solar_{}_R1M/'.format(solar,sp)
    order_by_order = False # turn off these models for the moment (may be more efficient in future)
    if order_by_order:
        model_dir += 'order_by_order/'

    # results file
    save_dir      = 'xcorr_result/'+'{}Solar_{}_R1M/'.format(solar,sp)
    simple        = True # turn on (true)/off simple pearsonr cross-correlation
    if simple:
        save_dir += 'pearsonr/'
        nam_res   = save_dir+'corr_velocity.pkl'
    else:
        save_dir += 'boucher/'
        nam_res   = save_dir+'boucher_corr_Kp_vsys.pkl'
        nam_fig   = save_dir+'Kp_vsys_map_ALLorders.png'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #name_model = "Model/Mod_boucher.txt"
    #name_wav   = "Model/Wave_boucher.txt"
    #filename   = sys.argv[1]  #"Results/New/Sep18/reduced_sep18.pkl"
    #nam_fig    = sys.argv[2] #"Results/New/Sep18/correl_sep18.png"
    #nam_res    = sys.argv[3] #"Results/New/Sep18/correl_sep18.pkl"


    Rs         = 261413.0   # Stellar radius [km]

    ### Velocimetric semi-amplitude
    Kpmin      = 0.0 #Jupiter
    Kpmax      = 280.0#Jupiter
    Nkp        = 100 ### Size of the grid
    Kp         = np.linspace(Kpmin,Kpmax,Nkp)

    ### Mid-transit planet RV [km/s]
    Vmin       = -40.0
    Vmax       =  40.0
    Nv         = 81
    Vsys       = np.linspace(Vmin,Vmax,Nv)




    ### READ data
    print("Read data from",filename)
    with open(filename,'rb') as specfile:
        orders,WW,Ir,T_obs,phase,window,berv,vstar,airmass,SN = pickle.load(specfile)
    nord     = len(orders)

    ### Select orders for the correlation
    ord_sel    = orders
    V_shift    = vstar - berv # think this has shape (nep), velocity to shift into stellar rest frame


    ### Select orders for the correlation
    ord_sel    = orders
    V_shift    = vstar - berv # think this has shape (nep), velocity to shift into stellar rest frame


    print(nord,"orders detected")
    list_ord = []
    for nn in range(nord):
        O        = Order(orders[nn])
        O.W_fin  = np.array(WW[nn],dtype=float)
        O.W_fin  /= 1e3 # hack to convert to um (make universal later)
        O.I_pca  = np.array(Ir[nn],dtype=float)
        O.SNR    = np.array(SN[nn],dtype=float)
        O.W_mean = O.W_fin.mean()

        if order_by_order:
            # load model for each order
            mod_file = model_dir + 'template_det' +str(orders[nn]) + '.pic'
            W_mod,T_depth = pickle.load(open(mod_file,'rb'))
            O.Wm     = W_mod
            O.Im     = T_depth
        list_ord.append(O)
    print("DONE\n")


    # Use the below for a single global model
    #W_mod,I_mod    = np.loadtxt(name_wav),np.loadtxt(name_model)
    #T_depth        = np.copy(I_mod)
    #T_depth        = 1 - (I_mod/(1e5))**(2) / Rs**(2)
    #maxf           = ndimage.maximum_filter(T_depth,size=10000)

    if not order_by_order:
        mod_file = model_dir + 'template_det1.pic'
        W_mod,T_depth = pickle.load(open(mod_file,'rb'))
        for kk,O in enumerate(list_ord):
            Wmin,Wmax = 0.95*O.W_fin.min(),1.05*O.W_fin.max()
            indm      = np.where((W_mod>Wmin)&(W_mod<Wmax))[0]
            W_sel     = W_mod[indm]
            O.Wm      = W_sel
            O.Im      = T_depth[indm]

    vtot = np.linspace(-200, 200, 100)

    if simple:
        print("\nRunning pearsonr cross-correlation.")
        # for now during testing phase - note these plots include out of transit epochs
        #----
        vsys_time,vsys_kp = simple_correlation(np.array(list_ord),window,phase,Kp,vtot,\
                            plot=True,savedir=save_dir)

        savedata = (vsys_time,vsys_kp)
        with open(nam_res, 'wb') as specfile:
            pickle.dump(savedata,specfile)
        print("DONE")
        #----
    else:
        ### Correlation
        ind_sel = []
        for kk,oo in enumerate(list_ord):
            if oo.number in ord_sel: ind_sel.append(kk)
        corr = compute_correlation(np.array(list_ord)[ind_sel],window,phase,Kp,Vsys,V_shift)

        #### Compute statistics and plot the map
        # Indicate regions to exclude when computing the NOISE level from the correlation map
        Kp_lim      = [80.0,160.0]   # Exclude this Kp range we
        Vsys_lim    = [-15.,15.]
        snrmap_fin  = get_snrmap(np.array(orders)[ind_sel],Kp,Vsys,corr,Kp_lim,Vsys_lim)
        sig_fin     = np.sum(np.sum(corr[:,:,ind_sel,:],axis=3),axis=2)/snrmap_fin



        ### Plot correlation + 1D cut
        K_cut   = 120.2
        V_cut   = 0.0
        ind_v   = np.argmin(np.abs(Vsys-V_cut))
        ind_k   = np.argmin(np.abs(Kp-K_cut))
        sn_map  = sig_fin
        sn_cutx = sn_map[:,ind_v]
        sn_cuty = sn_map[ind_k]
        cmap    = "gist_heat"

        ### Save data
        savedata = (Vsys,Kp,corr,sn_map)
        with open(nam_res, 'wb') as specfile:
            pickle.dump(savedata,specfile)
        print("DONE")

        plot_correlation_map(Vsys,Kp,sn_map,nam_fig,V_cut,K_cut,cmap,[],sn_cuty,20)
        #plot_correlation_map(Vsys,Kp,sn_map,nam_fig,K_cut,V_cut,cmap,sn_cutx,sn_cuty,20)





        ### Get and display statistics
        p_best,K_best,K_sup,K_inf,V_best,V_sup,V_inf = get_statistics(Vsys,Kp,sig_fin)
