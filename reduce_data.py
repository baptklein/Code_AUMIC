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
import time
from functions import *

############################# VERSION ADAPTED FOR IGRINS DATA

outroot = "Input_data/igrins/"

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



if mode_pca == "pca" or mode_pca == "PCA":
    outroot += "PCA/"
    nam_fin  = outroot+"reduced_1.pkl"
    nam_info = outroot+"info_1.dat"
    if not os.path.exists(outroot):
        os.makedirs(outroot)


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

    ### First we identify strong telluric lines and remove the data within these lines -- see Boucher+2021
    #W_cl,I_cl =  O.remove_tellurics(dep_min,thres_up)  ### Need a telluric spectrum


    #ind   = []
    #for nn in range(len(I_bl)):
    #    i = np.where(np.isfinite(I_bl[nn])==True)[0]
    #    ind.append(i)
    #r  = np.array(list(set.intersection(*map(set,ind))),dtype=int)
    #r  = np.sort(np.unique(r))

    W_cl,I_cl = np.copy(O.W_raw),np.copy(O.I_raw)+0.1



    ### If the order does not contain enough points, it is discarded
    if len(W_cl) < Npt_lim:
        print("ORDER",O.number,"(",O.W_mean,"nm) discarded (",len(W_cl)," pts remaining)")
        print("DISCARDED\n")
        ind_rem.append(nn)
        txt = str(O.number) + "   --\n"
        file.write(txt)
    else:
        print(len(O.W_raw)-len(W_cl),"pts removed from order",O.number,"(",O.W_mean,"nm) -- OK")

        if plot:
            nep,npix = I_cl.shape
            fig,axes = plt.subplots(2,1,figsize=(8,4))
            wmin,wmax = W_cl.min(),W_cl.max()
            dw = np.average(W_cl[1:]-W_cl[:-1])
            extent = (wmin - 0.5 * dw, wmax - 0.5 * dw, nep - 0.5, 0.5)
            xlabel = 'wavelength (nm)'
            mp1=axes[0].imshow(I_cl, extent=extent, interpolation='nearest', aspect='auto')
            fig.colorbar(mp1,ax=axes[0])
            axes[0].set_title('Order {}'.format(O.number))




        ### If the order is kept - Remove high-SNR out-of-transit reference spectrum
        ### Start by computing mean spectrum in the stellar rest frame
        V_cl      = c0*(W_cl/O.W_mean-1.)
        I_bary    = move_spec(V_cl,I_cl,V_corr,sig_g)  ## Shift to stellar rest frame
        I_med     = np.median(np.concatenate((I_bary[:n_ini],I_bary[n_end:]),axis=0),axis=0) ## Compute median out-of-transit
        I_med_geo = move_spec(V_cl,np.array([I_med]),-1.*V_corr,sig_g)  ## Move back ref spectrum to Geocentric frame
        I_sub1    = np.zeros(I_cl.shape)

        #plt.figure(figsize=(20,5))
        #for ii in range(len(I_cl)):
        #    plt.plot(W_cl,I_cl[ii])
        #plt.show()


        for kk in range(len(I_cl)):
            X          = np.array([np.ones(len(I_med_geo[kk])),I_med_geo[kk]],dtype=float).T
            p,pe       = LS(X,I_cl[kk])
            Ip         = np.dot(X,p)
            I_sub1[kk] = I_cl[kk]/Ip



        ### Then compute reference spectrum in the Geocentric frame
        I_med2  = np.median(np.concatenate((I_sub1[:n_ini],I_sub1[n_end:]),axis=0),axis=0)
        I_sub2  = np.zeros(I_sub1.shape)


        for kk in range(len(I_sub1)):
            #X          = np.array([np.ones(len(I_med2)),I_med2],dtype=float).T
            #p,pe       = LS(X,I_sub1[kk])
            #Ip         = np.dot(X,p)
            #plt.plot(I_sub1[kk])
            #plt.plot(Ip)
            #plt.plot(I_med2)
            #plt.show()
            I_sub2[kk] = I_sub1[kk]/I_med2

        #plt.plot(I_sub2[2])
        #I_med  = np.median(np.concatenate((I_cl[:n_ini],I_cl[n_end:]),axis=0),axis=0)
        #I_sub2 = I_cl/I_med


        ### Remove extremities to avoid interpolation errors
        W_sub = W_cl[N_bor:-N_bor]
        I_sub = I_sub2[:,N_bor:-N_bor]
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


        txt = str(O.number) + "  " + str(len(O.W_fin)) + "  " + str(np.mean(O.SNR)) + "  " + str(np.mean(O.SNR_mes)) + "  " + str(np.mean(O.SNR_mes_pca)) + "  " + str(n_com) + "\n"
        file.write(txt)

print("DATA REDUCTION DONE\n")
file.close()



### Plot final metrics -- RMS per spectrum in each order
print("PLOT METRICS")
orders_fin   = np.delete(orders,ind_rem)
list_ord_fin = np.delete(list_ord,ind_rem)
nam_fig      = outroot + "spectrum_dispersion.png"
plot_spectrum_dispersion(list_ord_fin,nam_fig)
print("DONE\n")

### Save data for correlation
print("\nData saved in",outroot+nam_fin)
Ir  = []
WW  = []
for nn in range(len(orders_fin)):
    O  = list_ord_fin[nn]
    WW.append(O.W_fin)
    Ir.append(O.I_pca)
savedata = (orders_fin,WW,Ir,T_obs,phase,window,berv,vstar,airmass,SN)
with open(nam_fin, 'wb') as specfile:
    pickle.dump(savedata,specfile)
print("DONE")

t1          = time.time()
print("DURATION:",(t1-t0)/60.,"min")
