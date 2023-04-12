#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Mar 2022
Edited in Jun 2022
@authors: Baptiste KLEIN, Florian DEBRAS & Annabella MEECH
"""
import matplotlib
matplotlib.use('Agg')
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
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", type=str, required=True,\
        help="select which data to analyse: IGRINS or SPIROU")
    parser.add_argument("--red-mode", type=str, default="PCA",\
        help="choose reduction type, options: PCA")
    parser.add_argument("--aligned", action="store_true", default=False,\
        help="use the aligned, wavelength recalibrated spectral matrix, \
         (for IGRINS data)")
    parser.add_argument("--masked", action="store_true", default=False,\
        help="use the masked spectra")
    parser.add_argument("--blazed", action="store_true", default=False,\
        help="for IGRINS use the blaze corrected data")
    parser.add_argument("--airmass", action="store_true", default=False,\
        help="use the airmass-detrended spectra")
    parser.add_argument("--deg-airmass", type=int, default=int(2),\
        help="degree of airmass-detrending")
    parser.add_argument("--residual-sampling", action='store_true', default=False,\
        help="residual sampling post PCA")
    parser.add_argument("--do-hipass", action='store_true', default=False,\
        help="hipass filter post PCA")
    parser.add_argument("--inject", action="store_true", default=False,\
        help="if true, use spectra with injected signal")
    parser.add_argument("--inj-Kp", type=float, default=0.0, \
        help="Kp velocity (km/s) at which to inject planet signal")
    parser.add_argument("--inj-vsys", type=float, default=0.0, \
        help="systemic velocity (km/s) at which to inject planet signal")
    parser.add_argument("--inj-amp", type=float, default=0.0, \
        help="scale factor of injected planet signal")
    parser.add_argument("--select-orders", action="store_true", default=False,\
        help="select only certain orders, otherwise use all, follow with orders argument")
    parser.add_argument("--orders", nargs='+', default=[], type=int, \
        help="name specific orders (numbers not indices), options 32--79,\
        just enter as integers with space between")
    parser.add_argument("--oot", action="store_true", default=False,\
        help="this option will reverse the transit window, and so cross-correlate with out of \
        transit frames only")
    parser.add_argument("--noLS", action="store_true", default=False,\
        help="this option will turn off the LS shift/stretch of the out-of-transit ref spec \
        in reduce_data.py")
    parser.add_argument("--Kp-lim", nargs=2, type=float, default=[60.,105.],\
        help="Indicate regions to exclude when computing the NOISE level from the correlation map")
    parser.add_argument("--Vsys-lim", nargs=2, type=float, default=[-15.,15.],\
        help="Indicate regions to exclude when computing the NOISE level from the correlation map,\
        negative argument to be fed in quotes and with space")
    parser.add_argument("--Mp", type=float, default=11.7,\
        help="Mass of planet in terms of Earth masses, selects model with which to cross-correlate\
        and/or that injected")
    parser.add_argument("--Rp", type=float, default=0.363,\
        help="Radius of planet in terms of Jupiter radii, selects model with which to cross-correlate\
        and/or that injected")



    args = parser.parse_args()

    data_dir  = 'Input_data/'
    if args.instrument == 'igrins' or args.instrument == 'IGRINS':
        data_dir   += 'igrins/'
        instrument = 'igrins'
        R          = 45000
    elif args.instrument == 'spirou' or args.instrument == 'SPIROU':
        data_dir   += 'spirou/'
        instrument = 'spirou'
        R          = 70000

    # model files
    species     = ['CO2']#,'CO','CO2','H2O','NH3'] # edit to include species in model ['CH4','CO','CO2','H2O','NH3']
    species.sort()
    sp          = '_'.join(i for i in species)
    Mp = args.Mp
    Rp = args.Rp
    order_by_order = False # turn off these models for the moment (may be more efficient in future)
    # binary model optional
    use_binary_mask = False
    if use_binary_mask:
        if args.inject:
            sys.exit('cannot inject binary mask, turn off')
        # load model
        if len(species)>1:
            sys.exit('cannot use binary mask for more than one species')
        filename = 'Models/hitemp_binary/HITEMP_binary_{}_3500_30000.npz'.format(species[0])
        d     = np.load(filename)
        T_depth  = d['mask']
        T_depth  = 1-T_depth
        T_depth -= 1
        W_mod    = d['wlens']*1e6 # convert to um
        print(np.min(W_mod))
        print(np.max(W_mod))
        binary = '_binary'

    else:

        solar       = '100x'
        CO_ratio    = '1.0'

        if order_by_order and instrument=='spirou':
            sys.exit('turn off order models for SPIRou')
        #if instrument=='igrins':
        #    model_dir   = 'pRT_models/' # works with pRT_make_spec.py
        #    model_dir  += 'aumicb_{}Solar_{}_R1M/'.format(solar,sp)
        #    if order_by_order:
        #        model_dir += 'order_by_order/'
        #elif instrument=='spirou':
        model_dir = 'Models/{:.3f}Mearth_{:.3f}Rjup/{}_metallicity_{}_CO_ratio/'.format(Mp,Rp,solar,CO_ratio)
        binary = ''

    if args.inject:
        print("loading in spectra with injected signal...")
        print("check input Kp-lim and Vsys-lim")
        data_dir += '{:.3f}Mearth_{:.3f}Rjup/'.format(Mp,Rp)
        data_dir += "inject_amp{:.1f}_Kp{:.1f}_vsys{:.2f}_{}/".format(args.inj_amp,\
        args.inj_Kp,args.inj_vsys,sp)
        text = 'injection {}x'.format(int(args.inj_amp))
    else:
        data_dir += "true_data/"
        text = 'true data'
    if args.blazed:
        data_dir += 'blazed/'
    if args.airmass:
        data_dir += 'airmass_deg{}/'.format(args.deg_airmass)
    if args.red_mode=='pca' or args.red_mode=='PCA':
        data_dir += 'PCA/'

    if args.residual_sampling:
        data_dir += 'residual_sampling/'
    if args.do_hipass:
        data_dir += 'hipass/'
    if args.noLS:
        data_dir += 'noLS/'

    if args.aligned:
        filename = "reduced_aligned.pkl"
        al       = "_aligned"
    else:
        filename = "reduced_1.pkl"
        al       = ""

    if args.masked:
        mk       = "_masked"
    else:
        mk       = ""
    if args.oot:
        oot      = True
        OOT      = "_oot"
    else:
        oot      = False
        OOT      = ""



    # results file
    if use_binary_mask:
        save_dir = 'xcorr_result/'+'{}/'.format(instrument)+'binary_mask/'
    else:
        save_dir    = 'xcorr_result/'+'{}/'.format(instrument)+'{:.3f}Mearth_{:.3f}Rjup/'.format(Mp,Rp)+'{}_metallicity_{}_CO_ratio/'.format(solar,CO_ratio)
    simple          = False # turn on (true)/off simple pearsonr cross-correlation
    plot_all_orders = True

    if args.inject:
        save_dir += "inject_amp{:.1f}_Kp{:.1f}_vsys{:.2f}_{}/".format(args.inj_amp,\
        args.inj_Kp,args.inj_vsys,sp)
    else:
        save_dir += "true_data/"
    if args.blazed:
        save_dir += "blazed/"
    if args.airmass:
        save_dir += 'airmass_deg{}/'.format(args.deg_airmass)
    if args.red_mode=='pca' or args.red_mode=='PCA':
        save_dir += 'PCA/'
    if args.residual_sampling:
        save_dir += 'residual_sampling/'
    if args.do_hipass:
        save_dir += 'hipass/'
    if args.noLS:
        save_dir += 'noLS/'
    if simple:
        save_dir += 'pearsonr/'
    else:
        save_dir += 'boucher/'

    #print(args.orders)
    if args.select_orders:
        if len(args.orders)>0:
            print('selecting orders')
            select_orders = args.orders
            _orders = '_'.join(str(i) for i in select_orders)
            nam_res   = save_dir+'corr_Kp_vsys_{}{}{}{}{}.pkl'.format(sp,binary,al,mk,OOT)
            nam_res2  = save_dir+'orders_{}/snmap_{}{}{}{}{}.pkl'.format(_orders,sp,binary,al,mk,OOT)
            nam_fig   = save_dir+'orders_{}/Kp_vsys_map_{}{}{}{}{}.png'.format(_orders,sp,binary,al,mk,OOT)
        else:
            sys.exit('specify orders to select if using select-orders argument')
    else:
        nam_res   = save_dir+'corr_Kp_vsys_{}{}{}{}{}.pkl'.format(sp,binary,al,mk,OOT)
        nam_res2  = save_dir+'snmap_{}{}{}{}{}.pkl'.format(sp,binary,al,mk,OOT)
        nam_fig   = save_dir+'Kp_vsys_map_ALLorders_{}{}{}{}{}.png'.format(sp,binary,al,mk,OOT)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.select_orders:
        if not os.path.exists(save_dir+'orders_{}/'.format(_orders)):
            os.makedirs(save_dir+'orders_{}/'.format(_orders))

    Kp_lim     = args.Kp_lim
    Vsys_lim   = args.Vsys_lim
    print(Kp_lim,Vsys_lim)


    Rs         = 261413.0   # Stellar radius [km]

    ### Velocimetric semi-amplitude
    Kpmin      = 0.0 #Jupiter
    Kpmax      = 150.0#Jupiter
    Nkp        = 151 ### Size of the grid
    Kp         = np.linspace(Kpmin,Kpmax,Nkp)

    ### Mid-transit planet RV [km/s]
    Vmin       = -30.0
    Vmax       =  30.0
    Nv         = 61
    Vsys       = np.linspace(Vmin,Vmax,Nv)




    ### READ data
    print("Read data from",data_dir+filename)
    with open(data_dir+filename,'rb') as specfile:
        A = pickle.load(specfile)
        if instrument=='igrins':
            if len(A)>16:
                orders,WW,Ir,T_obs,phase,window,berv,vstar,airmass,SN,SNR_mes,SNR_mes_pca,Imask,mask,MPC,NPC,std_ampl = A
            else:
                orders,WW,Ir,T_obs,phase,window,berv,vstar,airmass,SN,SNR_mes,SNR_mes_pca,Imask,mask,MPC,NPC = A

        elif instrument=='spirou' and args.airmass:
            if len(A)>16:
                orders,WW,Ir,T_obs,phase,window,berv,vstar,airmass,SN,SNR_mes,SNR_mes_pca,Imask,mask,MPC,NPC,std_ampl = A
            else:
                orders,WW,Ir,T_obs,phase,window,berv,vstar,airmass,SN,SNR_mes,SNR_mes_pca,Imask,mask,MPC,NPC = A
        else:
            orders,WW,Ir,T_obs,phase,window,berv,vstar,airmass,SN = A
    if oot:
        # reverse window, to only cross-correlate with out-of-transit frames
        # but no weightings for ingress/egress
        # only use purely out-of-transit
        window_new      = np.zeros_like(window)
        ind             = np.where(window==0.0)[0]
        window_new[ind] = 1.
        window          = window_new # overwrite

    nord     = len(orders)

    ### Select orders for the correlation
    if args.select_orders:
        ord_sel = select_orders
    else:
        ord_sel = orders
    V_shift    = vstar - berv # think this has shape (nep), velocity to shift into stellar rest frame
    if args.inject: V_shift    -= -4.71 # hack for the cross-correlation bc otherwise true vsys is added


    print(nord,"orders detected")
    list_ord = []
    for nn in range(nord):
        O        = Order(orders[nn])
        O.W_fin  = np.array(WW[nn],dtype=float)
        O.W_fin  /= 1e3 # hack to convert to um (make universal later)
        O.SNR    = np.array(SN[nn],dtype=float)
        O.W_mean = O.W_fin.mean()
        print(O.W_mean)
        if args.masked:
            O.I_pca = np.array(Imask[nn],dtype=float)
        else:
            O.I_pca  = np.array(Ir[nn],dtype=float)
        if order_by_order:
            # load model for each order
            mod_file = model_dir + 'template_det' +str(orders[nn]) + '.pic'
            W_mod,T_depth = pickle.load(open(mod_file,'rb'))
            O.Wm     = W_mod
            O.Im     = T_depth

        ### purge NaNs
        l = np.ones_like(O.W_fin,'bool')
        for ipix in range(len(O.I_pca[0])):
            if np.isnan(O.I_pca[:,ipix]).any():
                l[ipix] = False
        O.W_fin = O.W_fin[l]
        O.I_pca = O.I_pca[:,l]
        O.M_pca = MPC[nn]    # pca removed from the data
        O.ncom_pca = NPC[nn] # number of pca components removed
        list_ord.append(O)


    print("DONE\n")

    ind_sel = []
    for kk,oo in enumerate(list_ord):
        if oo.number in ord_sel: ind_sel.append(kk)
    # Use the below for a single global model
    #W_mod,I_mod    = np.loadtxt(name_wav),np.loadtxt(name_model)
    #T_depth        = np.copy(I_mod)
    #T_depth        = 1 - (I_mod/(1e5))**(2) / Rs**(2)
    #maxf           = ndimage.maximum_filter(T_depth,size=10000)

    if not order_by_order:
        print('loading {} model'.format(sp))
        #if instrument=='igrins':
        #    mod_file = model_dir + 'template_det1.pic'
        #    W_mod,T_depth = pickle.load(open(mod_file,'rb'))
        #elif instrument=='spirou':
        #mod_file = model_dir+'pRT_data_full_{}_R{}.dat'.format(sp,R)
        if not use_binary_mask:

            mod_file = model_dir+'pRT_data_full_{}.dat'.format(sp)
            W_mod = []
            T_depth = []
            with open(mod_file, 'r') as data:
                lines = data.readlines()
                data.close()
            for line in lines[4:]:
                v = line.split(' ')
                W_mod.append(float(v[0]))
                T_depth.append(float(v[1].split('\n')[0]))
            W_mod = np.array(W_mod)/1e3 # convert to um
            T_depth = np.array(T_depth)
            if args.inject:
                T_depth = (1 + args.inj_amp*(T_depth-1))
            T_depth -=1
        for kk,O in enumerate(list_ord):
            Wmin,Wmax = 0.95*O.W_fin.min(),1.05*O.W_fin.max()
            indm      = np.where((W_mod>Wmin)&(W_mod<Wmax))[0]
            O.Wm      = W_mod#indm]
            O.Im      = T_depth#[indm]

    vtot = np.linspace(-200, 200, 100)

    # check if cross-correlation already computed
    if Path(nam_res).is_file():
        with open(nam_res, 'rb') as specfile:
            Vsys,Kp,corr = pickle.load(specfile)
        print("cross-correlation result loaded")

    else:
        if simple:
            print("\nRunning pearsonr cross-correlation.")
            # for now during testing phase - note these plots include out of transit epochs
            # does this not work with order_by_order?
            #----
            vsys_time,vsys_kp = simple_correlation(np.array(list_ord),window,phase,Kp,vtot,\
                                plot=True,savedir=save_dir,oot=oot)

            savedata = (vsys_time,vsys_kp)
            with open(nam_res, 'wb') as specfile:
                pickle.dump(savedata,specfile)
            print("DONE")
            exit()
            #----
        else:
            # compute correlation for all orders, save, then later combine chosen orders for plots
            corr = compute_correlation(np.array(list_ord),window,phase,Kp,Vsys,V_shift,reprocess=False)

        ### Save data
        savedata = (Vsys,Kp,corr)
        with open(nam_res, 'wb') as specfile:
            pickle.dump(savedata,specfile)
        print("DONE")

    #### Compute statistics and plot the map
    snrmap_fin  = get_snrmap(np.array(orders)[ind_sel],Kp,Vsys,corr[:,:,ind_sel],Kp_lim,Vsys_lim)
    sig_fin     = np.sum(np.sum(corr[:,:,ind_sel,:],axis=3),axis=2)/snrmap_fin # axis 3 is time, axis 2 is order # does this line still work for one order?


    ### Plot correlation + 1D cut
    K_cut   = 83. # expected 83 km/s
    V_cut   = 0.0
    ind_v   = np.argmin(np.abs(Vsys-V_cut))
    ind_k   = np.argmin(np.abs(Kp-K_cut))
    sn_map  = sig_fin
    sn_cutx = sn_map[:,ind_v]
    sn_cuty = sn_map[ind_k]
    cmap    = "gist_heat"

    ### Save snr map
    with open(nam_res2, 'wb') as specfile:
        pickle.dump(sn_map,specfile)
    print("DONE")

    ### Get and display statistics
    #p_best,K_best,K_sup,K_inf,V_best,V_sup,V_inf = get_statistics(Vsys,Kp,sig_fin)

    if args.inject:
        V_cut = round(args.inj_vsys,2)
        K_cut = round(args.inj_Kp,2)
    else:
        print(V_cut)
        #V_cut = round(V_best,1)
        #K_cut = round(K_best,1)
    plot_correlation_map(Vsys,Kp,sn_map,nam_fig,V_cut,K_cut,cmap,[],sn_cuty,20,pointer=True,box=False,text=True,text_title='{} {} \n$K_p$ = {:.2f} km/s \n$V_s$ = {:.2f} km/s'.format(sp,text,K_cut,V_cut))
    if plot_all_orders:
        dir_ord = save_dir+'individual_orders_{}{}/'.format(sp,binary)
        if args.masked: dir_ord += 'masked/'
        if not os.path.exists(dir_ord):
            os.makedirs(dir_ord)

        for iord in range(len(list_ord)):
            O         = list_ord[iord]
            snmap_ord = get_snrmap(np.array(orders)[[iord]],Kp,Vsys,corr[:,:,[iord],:],Kp_lim,Vsys_lim)
            sigma_ord = np.sum(corr[:,:,iord,:],axis=2)/snmap_ord
            fig_ord   = dir_ord+'Kp_vsys_map_{}{}{}{}_ord{}_iord{}.png'.format(sp,al,mk,OOT,O.number,iord)
            plot_correlation_map(Vsys,Kp,sigma_ord,fig_ord,V_cut,K_cut,cmap,[],sn_cuty,20,pointer=True,box=False,text=True,text_title='{} {} \n$K_p$ = {:.2f} km/s \n$V_s$ = {:.2f} km/s'.format(sp,text,K_cut,V_cut))
    #plot_correlation_map(Vsys,Kp,sn_map,nam_fig,K_cut,V_cut,cmap,sn_cutx,sn_cuty,20)
