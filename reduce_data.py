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

############################# VERSION ADAPTED FOR IGRINS DATA
outroot = 'Input_data/'
instrument = 'igrins'

if instrument == 'IGRINS' or 'igrins':
    outroot += 'igrins/'
elif instrument =='SPIROU' or 'spirou':
    outroot += 'spirou/'

filename = outroot+"data_igrins.pkl" ### Name of the pickle file to read the data from

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

align    = True      # optionally align the spectra


### Data reduction parameters
dep_min  = 0.5       # remove all data when telluric relative absorption < 1 - dep_min
thres_up = 0.03      # Remove the line until reaching 1-thres_up
Npt_lim  = 2000       # If the order contains less than Npt_lim points, it is discarded from the analysis

### Interpolation parameters
sig_g    = 1.0                         ### STD of one SPIRou px in km/s
N_bor    = 15                           ### Nb of pts removed at each extremity (twice)

### Normalisation parameters
N_med    = 50                          ### Nb of points used in the median filter for the inteprolation
sig_out  = 5.0                          ### Threshold for outliers identification during normalisation process
deg_px   = 2                            ### Degree of the polynomial fit to the distribution of pixel STDs

### Parameters for detrending with airmass
det_airmass = False
deg_airmass = 2

### Parameters PCA
mode_pca    = "pca"                     ### "pca"/"PCA" or "autoencoder"
npca        = np.array(2*np.ones(len(orders)),dtype=int)      ### Nb of removed components
auto_tune   = True                             ### Automatic tuning of number of components based on white noise maps amplified by blaze

### Parameters for masking
fac       = 2.0 # factor of std at which to mask

if align:
    outroot += 'aligned/'
if det_airmass:
    outroot += 'airmass/'
if mode_pca == "pca" or mode_pca == "PCA":
    outroot += "PCA/"
nam_fin  = outroot+"reduced_1.pkl"
nam_info = outroot+"info_1.dat"
os.makedirs(outroot,exist_ok=True)
os.makedirs(outroot+'masked/',exist_ok=True)


### Create order objects
nord     = len(orders)
print(nord,"orders detected")
list_ord = []
for nn in range(nord):
    O        = Order(orders[nn])
    O.W_raw  = np.array(WW[nn],dtype=float)
    O.I_raw  = np.array(Ir[nn],dtype=float)
    #O.blaze  = np.array(blaze[nn],dtype=float)
    #O.I_atm  = np.array(Ia[nn],dtype=float)
    O.SNR    = np.array(SN[nn],dtype=float)
    O.W_mean = O.W_raw.mean()
    list_ord.append(O)
print("DONE\n")


ind_rem     = []
V_corr      = vstar - berv                  ### Geo-to-bary correction
n_ini,n_end = get_transit_dates(window)     ### Get transits start and end indices
c0          = Constants().c0
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
            plot=True
        else:
            plot=False
    O         = list_ord[nn]
    print("ORDER",O.number)
    print(O.W_mean)

    ### First we identify strong telluric lines and remove the data within these lines -- see Boucher+2021
    #W_cl,I_cl =  O.remove_tellurics(dep_min,thres_up)  ### Need a telluric spectrum


    #ind   = []
    #for nn in range(len(I_bl)):
    #    i = np.where(np.isfinite(I_bl[nn])==True)[0]
    #    ind.append(i)
    #r  = np.array(list(set.intersection(*map(set,ind))),dtype=int)
    #r  = np.sort(np.unique(r))

    W_cl,I_cl = np.copy(O.W_raw),np.copy(O.I_raw)+0.1
    nep,npix  = I_cl.shape

    # purge nans and negatives
    I_cl[I_cl<0] = 0.
    I_cl[np.isnan(I_cl)] = 0.


    ### If the order does not contain enough points, it is discarded
    if len(W_cl) < Npt_lim:
        print("ORDER",O.number,"(",O.W_mean,"nm) discarded (",len(W_cl)," pts remaining)")
        print("DISCARDED\n")
        ind_rem.append(nn)
        txt = str(O.number) + "   --\n"
        file.write(txt)
    else:
        print(len(O.W_raw)-len(W_cl),"pts removed from order",O.number,"(",O.W_mean,"nm) -- OK")

        if align:
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

        if plot:
            fig,axes = plt.subplots(2,1,figsize=(8,4))
            wmin,wmax = W_cl.min(),W_cl.max()
            dw = np.average(W_cl[1:]-W_cl[:-1])
            extent = (wmin - 0.5 * dw, wmax - 0.5 * dw, nep - 0.5, 0.5)
            xlabel = 'wavelength (nm)'
            mp1=axes[0].imshow(I_cl, extent=extent, interpolation='nearest', aspect='auto')
            fig.colorbar(mp1,ax=axes[0])
            axes[0].set_title('Order {}'.format(O.number))

        ### First compute reference spectrum in the Geocentric frame
        I_med2  = np.median(np.concatenate((I_cl[:n_ini],I_cl[n_end:]),axis=0),axis=0)
        I_sub2  = np.zeros(I_cl.shape)

        # for each epoch
        for kk in range(len(I_cl)):
            X          = np.array([np.ones(len(I_med2)),I_med2],dtype=float).T
            p,pe       = LS(X,I_cl[kk])
            Ip         = np.dot(X,p)
            I_sub2[kk] = I_cl[kk]/Ip#I_med2
        ### If the order is kept - Remove high-SNR(?) out-of-transit reference spectrum
        ### Start by computing mean spectrum in the stellar rest frame
        V_cl      = c0*(W_cl/O.W_mean-1.)
        I_bary    = move_spec(V_cl,I_sub2,V_corr,sig_g)  ## Shift to stellar rest frame
        I_med     = np.median(np.concatenate((I_bary[:n_ini],I_bary[n_end:]),axis=0),axis=0) ## Compute median out-of-transit
        I_med_geo = move_spec(V_cl,np.array([I_med]),-1.*V_corr,sig_g)  ## Move back ref spectrum to Geocentric frame
        I_sub1    = np.zeros(I_sub2.shape)


        # a stretch/shift of the stellar ref spec to each spectrum (then remove)?
        for kk in range(len(I_cl)):
            X          = np.array([np.ones(len(I_med_geo[kk])),I_med_geo[kk]],dtype=float).T
            p,pe       = LS(X,I_sub2[kk])
            Ip         = np.dot(X,p)
            I_sub1[kk] = I_sub2[kk]/Ip


        ### Remove extremities to avoid interpolation errors
        W_sub = W_cl[N_bor:-N_bor]
        I_sub = I_sub1[:,N_bor:-N_bor]
        ### END of STEP 1

        ### STEP 2 -- NORMALISATION AND OUTLIER REMOVAL
        W_norm1,I_norm1 = O.normalize(W_sub,I_sub,N_med,sig_out,N_bor)
        ### Correct for bad pixels

        W_norm2,I_norm2 = O.filter_pixel(W_norm1,I_norm1,deg_px,sig_out)
        ### END of STEP 2

        #plt.plot(I_norm2[2])



        ### STEP 3 -- DETREND WITH AIRMASS -- OPTIONAL
        if det_airmass:
            I_log           = np.log(I_norm2)
            I_det_log       = O.detrend_airmass(W_norm2,I_norm2,airmass,deg_airmass)
            I_det           = np.exp(I_det_log)
            O.I_fin         = I_det
        else:
            O.I_fin         = I_norm2



        O.W_fin  = np.copy(W_norm2)



        ### STEP 4 -- REMOVING CORRELATED NOISE -- PCA/AUTOENCODERS
        Il    = np.log(O.I_fin)
        im    = np.nanmean(Il)
        ist   = np.nanstd(Il)
        ff    = (Il - im)/ist


        XX    = np.where(np.isnan(O.I_fin[0]))[0]
        if len(XX) > 0:
            print("ORDER",O.number,"intractable: DISCARDED\n")
            ind_rem.append(nn)
        else:
            if mode_pca == "pca" or mode_pca == "PCA":

                if auto_tune: n_com = O.tune_pca(Nmap=5)
                else: n_com = npca[nn]
                pca   = PCA(n_components=n_com)
                x_pca = np.float32(ff)
                pca.fit(x_pca)
                principalComponents = pca.transform(x_pca)
                x_pca_projected = pca.inverse_transform(principalComponents)
                O.I_pca = np.exp((ff-x_pca_projected)*ist+im)
                NCF[nn] = n_com

                print(n_com,"PCA components discarded")


            ### ESTIMATES FINAL METRICS
            N_px          = 200
            indw          = np.argmin(np.abs(O.W_fin-O.W_fin.mean()))
            O.SNR_mes     = 1./np.std(O.I_fin[:,indw-N_px:indw+N_px],axis=1)
            O.SNR_mes_pca = 1./np.std(O.I_pca[:,indw-N_px:indw+N_px],axis=1)

            if plot:
                mp2=axes[1].imshow(O.I_pca, extent=extent, interpolation='nearest', aspect='auto')
                fig.colorbar(mp2,ax=axes[1])
                axes[1].set_xlabel(xlabel)
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
print("\nData saved in",outroot+nam_fin)
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
