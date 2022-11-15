#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Mar 2022
Edited in Jun 2022

@author: Baptiste KLEIN, Florian DEBRAS & Annabella MEECH
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.optimize import curve_fit
from scipy import interpolate
import time
from functions import *
import warnings
warnings.simplefilter('ignore',np.RankWarning)
############################# VERSION ADAPTED FOR IGRINS DATA
outroot    = 'Input_data/'
instrument = 'igrins' # models do not extend far enough for igrins
c0         = Constants().c0

if instrument == 'IGRINS' or instrument == 'igrins':
    outroot += 'igrins/'
elif instrument =='SPIROU' or instrument == 'spirou':
    outroot += 'spirou/'
else:
    sys.exit('choose spirou or igrins')


filename = outroot+"data_{}.pkl".format(instrument) ### Name of the pickle file to read the data from

### Read data in pickle format
### Namely:
#           - orders: List of orders -- absolute nbs
#           - WW:     Wavelength vector for each order [list obj (N_order,N_wav)]
#           - Ir:     Intensity values for each order [list of 2D arrays (N_order,N_obs,N_wav)]
#           - blaze:  Blaze values for each order [list of 2D arrays (N_order,N_obs,N_wav)]
#           - Ia:     Telluric spectra for each order [list of 2D arrays (N_order,N_obs,N_wav)]
#           - T_obs:  Observation dates [BJD]
#           - phase:  Planet orbital phase - centered on mid-transit
#           - window: Transit window (1 --> mid-transit; 0 --> out-of-transit)
#           - berv:   BERV values [km/s]
#           - vstar:  Stellar RV [km/s]
#           - airmass:Airmass values
#           - SN:     Signal-to-noise values for each order [N_order,N_obs]
print("Read data from",filename)

with open(filename,'rb') as specfile:
    A = pickle.load(specfile)
orders,WW,Ir,blaze,Ia,T_obs,phase,window,berv,vstar,airmass,SN = A

if instrument=='igrins' or instrument=='IGRINS':
    # I should save these in Ia under read_data.py
    skycalc_dir = outroot+'skycalc_models/'
    file_list = sorted(glob.glob(skycalc_dir+'skycalc_models_AU_MIC_Gemini_South_frame*.npz'))

    skycalc_models = [] # will be in order: nep, ndet, npix
    skycalc_wlens  = [] # same as models
    for ifile in range(len(file_list)):
        d = np.load(file_list[ifile],allow_pickle=True)
        skycalc_models.append(d['flux'])
        skycalc_wlens.append(d['wlens'])

### Injection parameters - optionally inject a planet model
inject   = False
inj_amp  = 1.
inj_Kp   = 83. #km/s 83km/s true_data
inj_vsys = 10.  #km/s -4.71 km/s true_data


### Data reduction parameters
align    = False      # optionally align the spectra
fitblaze = True       # optionally fit a blaze function to IGRINS spectra
dep_min  = 0.6        # remove all data when telluric relative absorption < 1 - dep_min
thres_up = 0.03       # Remove the line until reaching 1-thres_up
Npt_lim  = 200       # If the order contains less than Npt_lim points, it is discarded from the analysis
doLS     = True      # perform stretch/shift of reference stellar out-of-transit mean spectrum to each observed spectrum (ATM only turned off for spirou)

### Interpolation parameters
sig_g    = 1.0                         ### STD of one SPIRou px in km/s
N_bor    = 15                           ### Nb of pts removed at each extremity (twice)

### Normalisation parameters
N_med    = 50                          ### Nb of points used in the median filter for the interpolation
sig_out  = 5.0                          ### Threshold for outliers identification during normalisation process
deg_px   = 2                            ### Degree of the polynomial fit to the distribution of pixel STDs

### Parameters for detrending with airmass
det_airmass = True
deg_airmass = 2

### Parameters PCA
mode_pca    = "pca"                     ### "pca"/"PCA" or "autoencoder"
npca        = np.array(2*np.ones(len(orders)),dtype=int)      ### Nb of removed components
auto_tune   = True                             ### Automatic tuning of number of components based on white noise maps amplified by blaze
thr_pca     = 1.0                   ### PCA comp removed if eigenvalue larger than thr_pca*max(eigenvalue_noise)

sample_residuals = True  # optionally sample deep telluric residuals post PCA

### Parameters for masking
fac       = 2.0 # factor of std at which to mask

if inject:
    outroot += 'inject_amp{:.1f}_Kp{:.1f}_vsys{:.2f}'.format(inj_amp,inj_Kp,inj_vsys)
    # load model
    # model files
    species     = ['CH4'] # edit to include species in model ['CH4','CO','CO2','H2O','NH3']
    sp          = '_'.join(i for i in species)
    solar       = '1x'
    CO_ratio    = '1.0'
    model_dir = 'Models/{}_metallicity_{}_CO_ratio/'.format(solar,CO_ratio)
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
    W_mod    = np.array(W_mod)#/1e3 # convert to um
    T_depth  = np.array(T_depth)
    print(W_mod)
    mod_func = interpolate.interp1d(W_mod,T_depth)
    outroot += '_{}/'.format(sp)
else:
    outroot += 'true_data/'
if align:
    outroot += 'aligned/'
if fitblaze:
    outroot += 'blazed/'
if det_airmass:
    outroot += 'airmass_deg{}/'.format(deg_airmass)
if mode_pca == "pca" or mode_pca == "PCA":
    outroot += "PCA/"
if sample_residuals:
    outroot += 'residual_sampling/'
if not doLS:
    outroot += 'noLS/'
nam_fin  = outroot+"reduced_1.pkl"
nam_info = outroot+"info_1.dat"
os.makedirs(outroot,exist_ok=True)
os.makedirs(outroot+'masked/',exist_ok=True)


ind_rem     = []
V_corr      = vstar - berv                  ### Geocentric-to-stellar rest frame correction
n_ini,n_end = get_transit_dates(window)     ### Get transits start and end indices


### Create order objects
nord     = len(orders)
print(nord,"orders detected")
list_ord = []
for nn in range(nord):
    O        = Order(orders[nn])
    O.W_raw  = np.array(WW[nn],dtype=float)
    print(O.W_raw.min())
    if inject:
        print('\n injecting a model of {}'.format(sp))
        print('\n at a Kp: {} km/s, vsys: {} km/s'.format(inj_Kp,inj_vsys))
        print('\n model scaled by a factor of {}'.format(inj_amp))
        # get wlens in planet rest frame
        vp           = inj_vsys - berv + inj_Kp*np.sin(2*np.pi*phase)
        vp           = vp[n_ini:n_end]
        shift_fac    = 1.0 / (1.0 + vp/c0)
        wlens_planet = O.W_raw[None,:] * shift_fac[:,None]
        model_prep   = mod_func(wlens_planet)
        flux         = np.array(Ir[nn],dtype=float)
        flux[n_ini:n_end,:] *= (1 + inj_amp*(model_prep-1))
        O.I_raw      = flux
    else:
        O.I_raw  = np.array(Ir[nn],dtype=float)
    #O.blaze  = np.array(blaze[nn],dtype=float)
    #O.I_atm  = np.array(Ia[nn],dtype=float)
    O.SNR    = np.array(SN[nn],dtype=float)
    O.W_mean = O.W_raw.mean()
    if instrument=='igrins' or instrument=='IGRINS':
        # currently don't have skycalc models for spirou
        atm  = []
        watm = []
        for iep in range(len(O.I_raw)):
            atm.append(skycalc_models[iep][nn])
            watm.append(skycalc_wlens[iep][nn])
        O.I_atm = np.array(atm)
        O.W_atm = np.array(watm) # shape nep,npix but should be same in each ep

    list_ord.append(O)
print("DONE\n")

t0          = time.time()
NCF         = np.zeros(nord)


#### Main reduction
print("START DATA REDUCTION")
file = open(nam_info,"w")
plot_all_orders = True
if not plot_all_orders:
    plot_ord = 10 # pick an example order to plot
for nn in range(nord):
    if plot_all_orders:
        plot=True
    else:
        if nn==plot_ord:
            plot=False
        else:
            plot=False
    O         = list_ord[nn]
    print("ORDER",O.number)
    print(O.W_mean)
    if nn==0:
        print('discard first order')
        ind_rem.append(nn)
        txt = str(O.number) + "   --\n"
        file.write(txt)
        continue

    ### First we identify strong telluric lines and remove the data within these lines -- see Boucher+2021
    #
    #if instrument=='igrins' or instrument=='IGRINS':
    #    W_cl,I_cl =  O.remove_tellurics(dep_min,thres_up)  ### Need a telluric spectrum
    #else:
    W_cl,I_cl = np.copy(O.W_raw),np.copy(O.I_raw)

    #if instrument=='igrins' or instrument=='IGRINS':
    #    I_cl+=0.1 # otherwise code ride loads of orders


    if len(W_cl)>0:
        if instrument=='igrins' or instrument=='IGRINS':
            # mask low flux before fitting blaze
            I_mean = np.mean(I_cl,axis=0)
            ind_low = np.where(I_mean<=0.15)[0]
            I_cl[:,ind_low] = np.nan
            W_cl[ind_low] = np.nan

        # purge nans and negatives
        nep,npix = I_cl.shape
        ind   = []
        for iep in range(nep):
            i = np.where(np.isfinite(I_cl[iep])==True)[0]
            ind.append(i)
        r       = np.array(list(set.intersection(*map(set,ind))),dtype=int)
        r       = np.sort(np.unique(r))
        I_cl    = I_cl[:,r]
        W_cl    = W_cl[r]
        nep,npix = I_cl.shape
        #ind   = []
        #for iep in range(nep):
        #    i = np.where(I_cl[iep]>=0.0)[0]
        #    ind.append(i)
        #r       = np.array(list(set.intersection(*map(set,ind))),dtype=int)
        #r       = np.sort(np.unique(r))
        #I_cl    = I_cl[:,r]
        #W_cl    = W_cl[r]
    else:
        print("ORDER",O.number,"(",O.W_mean,"nm) discarded (0 pts remaining)")
        print("DISCARDED\n")
        ind_rem.append(nn)
        txt = str(O.number) + "   --\n"
        file.write(txt)

    ### If the order does not contain enough points, it is discarded
    if len(W_cl) < Npt_lim:


        print("ORDER",O.number,"(",O.W_mean,"nm) discarded (",len(W_cl)," pts remaining)")
        print("DISCARDED\n")
        ind_rem.append(nn)
        txt = str(O.number) + "   --\n"
        file.write(txt)
    else:
        print(len(O.W_raw)-len(W_cl),"pts removed from order",O.number,"(",O.W_mean,"nm) -- OK")

        nep,npix  = I_cl.shape
        if plot:
            fig,axes = plt.subplots(3,1,figsize=(8,8))
            #fig,axes = plt.subplots(2,1,figsize=(8,4))
            wmin,wmax = W_cl.min(),W_cl.max()
            dw = np.average(W_cl[1:]-W_cl[:-1])
            extent = (wmin - 0.5 * dw, wmax - 0.5 * dw, nep - 0.5, 0.5)
            xlabel = 'wavelength (nm)'
            mp1=axes[0].imshow(I_cl, extent=extent, interpolation='nearest', aspect='auto')
            fig.colorbar(mp1,ax=axes[0])
            axes[0].set_title('Order {}'.format(O.number))

        if align:
            # should move this before masking low flux
            # WOULD NEED TO PROPAGATE UNCERTAINTIES IF NEEDED FURTHER DOWN THE LINE
            # crop edge of order
            I_cl[:,:100] = 0.
            I_cl[:,-100:] = 0.
            # use brightest spectrum as reference
            b  = np.nanmean(I_cl,axis=1)
            ib = np.argsort(b)[-1]
            ref_spec = I_cl[ib,:]
            m  = np.isfinite(ref_spec)

            spec_aligned = np.zeros_like(I_cl)
            for iep in range(nep):
                spec_to_correct = I_cl[iep,:]
                l = np.isfinite(spec_to_correct)
                q = l*m # remove NaNs
                cs_data = interpolate.splrep(W_cl[q],spec_to_correct[q],s=0.0)
                try:
                    yy = ref_spec[q]
                    popt,pconv = curve_fit(lambda x,aa,bb: stretch_shift(x,cs_data,aa,bb),W_cl[q],yy,p0=np.array([1.,0]))
                    #popt,pconv = curve_fit(stretch_shift,W_cl[q],,p0=np.array([1.,0.]))
                    spec_refit = stretch_shift(W_cl,cs_data,*popt)
                    spec_refit[~q] = np.nan # maintain nans
                except:
                    print('Alignment failed for order {}, exposure {}'.format(O.number,iep))
                    print('optimal params not found, leave spectrum as is')
                    spec_refit = spec_to_correct
                if np.isnan(spec_refit).any():
                    sys.exit('order {} ep {} contains nans'.format(nn,nep))
                spec_aligned[iep] = spec_refit

                # plot an example
                if iep==20 and nn==10:
                    plt.figure(figsize=(15,8))
                    plt.plot(W_cl, ref_spec,color='black',label='Reference Spectrum')
                    plt.plot(W_cl, spec_to_correct,color='red',label='Original Spectrum')
                    plt.plot(W_cl, spec_refit, color='C0',alpha=0.8, label='Wavelength Corrected Spectrum')
                    plt.legend(frameon=False)
                    plt.tight_layout()
                    plt.savefig(outroot+"example_alignment_order{}_exp{}.png".format(O.number,nep))

            # purge nans and negatives
            spec_aligned[spec_aligned<0] = 0.
            spec_aligned[np.isnan(spec_aligned)] = 0.
            I_cl = spec_aligned



        ### STEP 1 -- remove master out-of-transits
        # First in Earth frame, then stellar for IGRINS (opposite for SPIRou)
        if instrument == 'IGRINS' or instrument == 'igrins':
            # first fit a blaze function
            # try fitting blaze without these low flux regions
            if fitblaze:
                blaze = []
                maxrms = 0.005
                # fit blaze for each epoch
                for iep in range(nep):
                    test_flux = I_cl[iep]
                    test_wlens = np.copy(W_cl)
                    #plt.plot(test_wlens,test_flux/np.max(test_flux))

                    mask = np.ones(len(test_flux),'bool')
                    normspec = test_flux/np.max(test_flux)
                    curcall = 0
                    residrms = 1
                    numcall = 10
                    while ((curcall < numcall) and (residrms > maxrms)):
                        #print('On iteration {} of {}'.format(curcall,numcall))
                        #with warnings.simplefilter('ignore',np.RankWarning):
                        z,mask,residrms,test_wlens,normspec = O.fit_blaze(test_wlens, normspec, maxrms,
                                         numcalls=numcall, curcall=curcall,verbose=False,showplot=False)
                        #mask = ~a
                        curcall +=1
                    wl = W_cl
                    cfit = np.poly1d(z)
                    #plt.plot(wl,test_flux/np.max(test_flux))
                    #plt.plot(wl,cfit(wl))
                    #plt.title(iep)
                    #plt.show()
                    blaze.append(cfit(wl))
                I_cl = I_cl/blaze



            ### First compute reference spectrum in the Geocentric frame
            #I_med2  = np.median(np.concatenate((I_cl[:n_ini],I_cl[n_end:]),axis=0),axis=0)
            #I_sub2  = np.zeros(I_cl.shape)

            # for each epoch
            #for kk in range(len(I_cl)):
            #    X          = np.array([np.ones(len(I_med2)),I_med2],dtype=float).T
            #    p,pe       = LS(X,I_cl[kk])
            #    Ip         = np.dot(X,p)
            #    I_sub2[kk] = I_cl[kk]/Ip
            ### Now mean spectrum in the stellar rest frame
            #V_cl      = c0*(W_cl/O.W_mean-1.)
            #I_bary    = move_spec(V_cl,I_sub2,V_corr,sig_g)  ## Shift to stellar rest frame
            #I_med     = np.median(np.concatenate((I_bary[:n_ini],I_bary[n_end:]),axis=0),axis=0) ## Compute median out-of-transit
            #I_med_geo = move_spec(V_cl,np.array([I_med]),-1.*V_corr,sig_g)  ## Move back ref spectrum to Geocentric frame
            #I_sub1    = np.zeros(I_sub2.shape)

            # a stretch/shift of the stellar ref spec to each spectrum (then remove)
            #for kk in range(len(I_cl)):
            #    X          = np.array([np.ones(len(I_med_geo[kk])),I_med_geo[kk]],dtype=float).T
            #    p,pe       = LS(X,I_sub2[kk])
            #    Ip         = np.dot(X,p)
            #    I_sub1[kk] = I_sub2[kk]/Ip

            ### Remove extremities to avoid interpolation errors
            W_sub = W_cl[N_bor:-N_bor]
            I_sub = I_cl[:,N_bor:-N_bor]#I_sub1[:,N_bor:-N_bor]

        elif instrument =='SPIROU' or instrument=='spirou':
            ### If the order is kept - Remove high-SNR(?) out-of-transit reference spectrum
            ### Start by computing mean spectrum in the stellar rest frame
            V_cl      = c0*(W_cl/O.W_mean-1.)
            I_bary    = move_spec(V_cl,I_cl,V_corr,sig_g)  ## Shift to stellar rest frame, result from move_spec will now contain nans
            I_med     = np.median(np.concatenate((I_bary[:n_ini],I_bary[n_end:]),axis=0),axis=0) ## Compute median out-of-transit
            I_med_geo = move_spec(V_cl,np.array([I_med]),-1.*V_corr,sig_g,fv=np.nan)  ## Move back ref spectrum to Geocentric frame
            I_sub1    = np.zeros(I_cl.shape) + np.nan

            for kk in range(len(I_cl)):
                if doLS:
                    l          = np.isfinite(I_med_geo[kk])
                    X          = np.array([np.ones(len(I_med_geo[kk][l])),I_med_geo[kk][l]],dtype=float).T
                    p,pe       = LS(X,I_cl[kk][l])
                    Ip         = np.dot(X,p)
                    I_sub1[kk][l] = I_cl[kk][l]/Ip #NANs from move_spec interpolation maintained
                else:
                    I_sub1[kk] = (I_cl[kk]/I_med_geo[kk]) # don't perform stretch/shift

            ### First compute reference spectrum in the Geocentric frame
            I_med2  = np.nanmedian(np.concatenate((I_sub1[:n_ini],I_sub1[n_end:]),axis=0),axis=0)
            I_sub2  = np.zeros(I_sub1.shape) + np.nan

            for kk in range(len(I_cl)):
                I_sub2[kk] = I_sub1[kk]/I_med2

            ### Remove extremities to avoid interpolation errors
            W_sub = W_cl[N_bor:-N_bor]
            I_sub = I_sub1[:,N_bor:-N_bor]#I_sub = I_sub2[:,N_bor:-N_bor]

        ### purge NaNs
        l = np.ones_like(W_sub,'bool')
        for ipix in range(len(I_sub[0])):
            if np.isnan(I_sub[:,ipix]).any():
                l[ipix] = False
        W_sub = W_sub[l]
        I_sub = I_sub[:,l]
        print('after blaze {}'.format(I_sub.shape))
        ### END of STEP 1
        print('after blaze {}'.format(np.isnan(I_sub.any())))
        ### STEP 2 -- NORMALISATION AND OUTLIER REMOVAL
        if fitblaze:
            W_norm1,I_norm1 = W_sub,I_sub
        else:
            W_norm1,I_norm1 = O.normalize(W_sub,I_sub,N_med,sig_out,N_bor)
        ### Correct for bad pixels

        W_norm2,I_norm2 = O.filter_pixel(W_norm1,I_norm1,deg_px,sig_out)
        ### END of STEP 2
        print('after normalisation {}'.format(np.isnan(I_norm2.any())))
        #plt.plot(I_norm2[2])
        ### STEP 3 -- DETREND WITH AIRMASS -- OPTIONAL

        ind_flag = []
        for uu in range(len(I_norm2)):
            iiii = np.where(I_norm2[uu]<0.05)[0]
            ind_flag.append(iiii)
        ind_flag = np.sort(np.unique(np.concatenate(ind_flag)))
        I_norm2  = np.delete(I_norm2,ind_flag,axis=1)
        W_norm2  = np.delete(W_norm2,ind_flag)
        print('after filters {}'.format(I_norm2.shape))
        if det_airmass:
            I_log           = np.log(I_norm2)
            #ind_flag = []
            #for iep in range(nep):
            #    iiii = np.where(I_log[iep]==0.)[0]# just happens to be flux =1. so log flux 0.
            #    ind_flag.append(iiii)
            #ind_flag = np.sort(np.unique(np.concatenate(ind_flag)))
            #print(ind_flag)
            #I_log = np.delete(I_log,ind_flag,axis=1) # for now just delete these values
            #W_norm2 = np.delete(W_norm2,ind_flag)
            print('after log {}'.format(np.isnan(I_log).any()))
            #print(W_norm2)
            #print(I_log)
            Il              = O.detrend_airmass(W_norm2,I_log,airmass,deg_airmass)
            O.I_fin         = np.exp(Il)
        else:
            O.I_fin         = I_norm2
            Il              = np.log(I_norm2)
        O.W_fin  = W_norm2
        print('after airmass detrending {}'.format(O.I_fin.shape))
        ### Removing the NaNs in Il
        #ind   = []
        #for uu in range(len(Il)):
        #    i = np.where(np.isfinite(Il[uu])==True)[0]
        #    ind.append(i)
        #r       = np.array(list(set.intersection(*map(set,ind))),dtype=int)
        #r       = np.sort(np.unique(r))
        #Il      = Il[:,r]
        #O.W_fin = O.W_fin[r]
        #O.I_fin = O.I_fin[:,r]
        #ff      = Il

        # remove NaNs
        ind_nan = []
        for uu in range(len(Il)):
            i = np.where(np.isnan(Il[uu]))[0]
            ind_nan.append(i)
        ind_nan = np.sort(np.unique(np.concatenate(ind_nan)))
        Il = np.delete(Il,ind_nan,axis=1)
        O.I_fin = np.delete(O.I_fin,ind_nan,axis=1)
        O.W_fin = np.delete(O.W_fin,ind_nan,axis=0)
        print('after nan filtering {}'.format(O.I_fin.shape))

        if len(O.W_fin)-len(r) > 0: print(len(O.W_fin)-len(r),"points rejected")

        ### STEP 4 -- REMOVING CORRELATED NOISE -- PCA/AUTOENCODERS
        Il    = np.log(O.I_fin)
        im  = np.nanmean(Il,axis=0)
        ist = np.nanstd(Il,axis=0)
        ff  = np.copy(Il)#(Il-im)/ist

        ind_nan = []
        for iep in range(len(ist)):
            # FOR NOW, mask columns with zero standard deviation
            if ist[iep]==0.:
                ind_nan.append(iep)
        ff  = np.delete(ff,ind_nan,axis=1)
        im  = np.delete(im,ind_nan,axis=0)
        ist = np.delete(ist,ind_nan,axis=0)
        im  = np.tile(im,(len(ff),1))
        ist = np.tile(ist,(len(ff),1))
        O.I_fin = np.delete(O.I_fin,ind_nan,axis=1)
        O.W_fin = np.delete(O.W_fin,ind_nan,axis=0)
        ff  = (ff-im)/ist


        #print('a:{}'.format(np.isnan(O.I_fin).any()))
        XX    = np.where(np.isnan(O.I_fin))[0]
        #print(XX)
        if len(XX) > 0:
            print("ORDER",O.number,"intractable: DISCARDED\n")
            ind_rem.append(nn)
        else:
            if mode_pca == "pca" or mode_pca == "PCA":

                if auto_tune: n_com = O.tune_pca(Nmap=5,thr=thr_pca)
                else: n_com = npca[nn]


                pca   = PCA(n_components=n_com)
                x_pca = np.float32(ff)
                pca.fit(x_pca)
                principalComponents = pca.transform(x_pca)
                x_pca_projected = pca.inverse_transform(principalComponents)
                O.I_pca = np.exp((ff-x_pca_projected)*ist+im) - 1.0
                NCF[nn] = n_com

                print(n_com,"PCA components discarded")

                if plot:
                    wmin,wmax = O.W_fin.min(),O.W_fin.max()
                    dw = np.average(O.W_fin[1:]-O.W_fin[:-1])
                    extent = (wmin - 0.5 * dw, wmax - 0.5 * dw, nep - 0.5, 0.5)
                    mp2=axes[1].imshow(O.I_pca, extent=extent, interpolation='nearest', aspect='auto')
                    fig.colorbar(mp2,ax=axes[1])
            # RESIDUAL SAMPLING OF DEEP TELLURIC LINES
            if sample_residuals:
                O.I_pca = O.telluric_residual_sampling(O.W_fin,O.I_pca)

            ### ESTIMATES FINAL METRICS
            N_px          = 200
            indw          = np.argmin(np.abs(O.W_fin-O.W_fin.mean()))
            O.SNR_mes     = 1./np.std(O.I_fin[:,indw-N_px:indw+N_px],axis=1)
            O.SNR_mes_pca = 1./np.std(O.I_pca[:,indw-N_px:indw+N_px],axis=1)

            if plot:
                wmin,wmax = O.W_fin.min(),O.W_fin.max()
                dw = np.average(O.W_fin[1:]-O.W_fin[:-1])
                extent = (wmin - 0.5 * dw, wmax - 0.5 * dw, nep - 0.5, 0.5)
                mp2=axes[2].imshow(O.I_pca, extent=extent, interpolation='nearest', aspect='auto')
                fig.colorbar(mp2,ax=axes[2])
                axes[2].set_xlabel(xlabel)
                plt.tight_layout()
                plt.savefig(outroot+"pca_reduced_order{}.png".format(O.number))


            ### POST-PCA MASKING OF NOISY COLUMNS
            if plot:
                fig,axes = plt.subplots(2,1,figsize=(8,4))
                wmin,wmax = W_cl.min(),W_cl.max()
                dw = np.average(W_cl[1:]-W_cl[:-1])
                extent = (wmin - 0.5 * dw, wmax - 0.5 * dw, nep - 0.5, 0.5)
                xlabel = 'wavelength (nm)'
                mp1=axes[0].imshow(O.I_pca, extent=extent, interpolation='nearest', aspect='auto')
                fig.colorbar(mp1,ax=axes[0])
                axes[0].set_title('Order {}'.format(O.number))

            # identify residual tellurics
            std       = np.std(O.I_pca,axis=0)
            thr       = fac*np.nanmedian(std)
            l         = std>thr
            l         = np.tile(l[None,:],(nep,1))
            I_mask    = np.copy(O.I_pca)
            I_mask[l] = np.nan
            O.I_mask  = I_mask
            mk        = np.isnan(I_mask)
            O.mask    = mk

            if plot:
                mp2=axes[1].imshow(O.I_mask, extent=extent, interpolation='nearest', aspect='auto')
                fig.colorbar(mp2,ax=axes[1])
                axes[1].set_xlabel(xlabel)
                plt.tight_layout()
                plt.savefig(outroot+"masked/masked_order{}.png".format(O.number))
                plt.close()

        txt = str(O.number) + "  " + str(len(O.W_fin)) + "  " + str(np.mean(O.SNR)) + "  " + str(np.mean(O.SNR_mes)) + "  " + str(np.mean(O.SNR_mes_pca)) + "  " + str(n_com) + "\n"
        if inject:
            txt += "injected model found at: {}, Kp: {}, vsys: {}, inj-amp: {}".format(mod_file,inj_Kp,inj_vsys,inj_amp)
        file.write(txt)

print("DATA REDUCTION DONE\n")
file.close()



### Plot final metrics -- RMS per spectrum in each order
print("PLOT METRICS")
orders_fin   = np.delete(orders,ind_rem)
list_ord_fin = np.delete(list_ord,ind_rem)
nam_fig      = outroot + "spectrum_dispersion.png"
plot_spectrum_dispersion(list_ord_fin,nam_fig,instrument)
print("DONE\n")

### Save data for correlation
print("\nData saved in",nam_fin)
Ir    = []
WW    = []
Imask = []
mask  = []
SNR_mes = []
SNR_mes_pca = []
for nn in range(len(orders_fin)):
    O  = list_ord_fin[nn]
    WW.append(O.W_fin)
    Ir.append(O.I_pca)
    Imask.append(O.I_mask)
    mask.append(O.mask)
    SNR_mes.append(O.SNR_mes)
    SNR_mes_pca.append(O.SNR_mes_pca)
savedata = (orders_fin,WW,Ir,T_obs,phase,window,berv,vstar,airmass,SN,SNR_mes,SNR_mes_pca,Imask,mask)
with open(nam_fin, 'wb') as specfile:
    pickle.dump(savedata,specfile)
print("DONE")

t1          = time.time()
print("DURATION:",(t1-t0)/60.,"min")
