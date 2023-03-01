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
from astropy.convolution import Gaussian1DKernel, convolve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.ndimage import percentile_filter,maximum_filter
import time
from functions import *
import warnings
warnings.simplefilter('ignore',np.RankWarning)
############################# VERSION ADAPTED FOR IGRINS DATA
outroot    = 'Input_data/'
instrument = 'spirou' # models do not extend far enough for igrins
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

# I should save these in Ia under read_data.py
skycalc_dir = outroot+'skycalc_models/'
if instrument=='igrins' or instrument=='IGRINS':
    mod_add  = 'Gemini_South'
    R        = 45000
elif instrument=='spirou' or instrument=='SPIROU':
    mod_add  = 'Canada-France-Hawaii_Telescope'
    R        = 70000
file_list = sorted(glob.glob(skycalc_dir+'skycalc_models_AU_MIC_{}_frame*.npz'.format(mod_add)))

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
inj_vsys = -4.71  #km/s -4.71 km/s true_data



### Data reduction parameters
StarRotator = False      # optionally remove StarRotator model at the beginning of reduction
phoenix     = True       # use phoenix option of StarRotator, if False use Kurucz
align       = False      # optionally align the spectra
fitblaze    = False       # optionally fit a blaze function to IGRINS spectra
dep_min     = 0.7        # remove all data when telluric relative absorption < 1 - dep_min
thres_up    = 0.1       # Remove the line until reaching 1-thres_up
Npt_lim     = 200       # If the order contains less than Npt_lim points, it is discarded from the analysis
doLS        = False      # perform stretch/shift of reference stellar out-of-transit mean spectrum to each observed spectrum (ATM only turned off for spirou)

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

# post-processing
sample_residuals = True  # optionally sample deep telluric residuals post PCA
do_hipass        = False

if fitblaze and instrument=='spirou':
    print('fitblaze not set up for spirou')
    fitblaze = False

### Parameters for masking
fac       = 1.8 # factor of std at which to mask

if inject:
    outroot += 'inject_amp{:.1f}_Kp{:.1f}_vsys{:.2f}'.format(inj_amp,inj_Kp,inj_vsys)
    # load model
    # model files
    species     = ['CO'] # edit to include species in model ['CH4','CO','CO2','H2O','NH3']
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
    W_mod    = np.array(W_mod) # nm
    T_depth  = np.array(T_depth)
    mod_func = interpolate.interp1d(W_mod,T_depth)
    outroot += '_{}/'.format(sp)
else:
    outroot += 'true_data/'
if StarRotator:
    if phoenix: ll = 'phoenix'
    else: ll = 'kurucz'
    outroot += 'StarRotator_{}/'.format(ll)

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
if do_hipass:
    outroot += 'hipass/'
if not doLS:
    outroot += 'noLS/'
nam_fin  = outroot+"reduced_1.pkl"
nam_info = outroot+"info_1.dat"
os.makedirs(outroot,exist_ok=True)
os.makedirs(outroot+'masked/',exist_ok=True)


ind_rem     = []
V_corr      = vstar - berv                  ### Geocentric-to-stellar rest frame correction
n_ini,n_end = get_transit_dates(window)     ### Get transits start and end indices

file = open(nam_info,"w")
### Create order objects
nord     = len(orders)
print(nord,"orders detected")
list_ord = []
for nn in range(nord):
    O        = Order(orders[nn])
    O.W_raw  = np.array(WW[nn],dtype=float) # nm
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
    O.blaze  = np.array(blaze[nn],dtype=float)
    O.SNR    = np.array(SN[nn],dtype=float)
    O.W_mean = O.W_raw.mean()
    if instrument=='spirou':
        O.I_atm  = np.array(Ia[nn],dtype=float)
    else:
        # skycalc models for telluric reference
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

if StarRotator:
    print("Uploading StarRotator model, convolving and interpolating\n")
    # upload model
    if phoenix: ll = '0mu'
    else:       ll = '20mu' # adjust number of mu angles here
    with open('../../StarRotator/AUMIC_{}_StarRotator_{}.pkl'.format(instrument,ll),'rb') as specfile:
        SRwl,SRspectra,SRstellar_spectrum,SRlightcurve,SRmask = pickle.load(specfile)
    # convolve to instrument resolution
    dv = c0/R/1000. # in km/s

    R_model = (SRwl[1:-1]+SRwl[0:-2])/(SRwl[1:-1]-SRwl[0:-2])/2
    sigma  = np.median(R_model)/R
    if sigma<1: sigma = 1 # otherwise 'increasing' resolution
    kernel = Gaussian1DKernel(stddev=sigma)

    for iord in range(nord):
        O        = list_ord[iord]
        WW_ord   = O.W_raw
        I_ord    = O.I_raw
        nep,npix = I_ord.shape

        # Shift PHOENIX model from stellar rest frame -> Doppler shift (and interpolate) to Earth rest frame
        # move W_cl Earth rest wavelengths to stellar rf
        WW_star = WW_ord[None,:] / ( 1 + V_corr[:,None]/c0)
        ## Evaluate StarRotator models in Earth rest frame for each order, convolved to SPIRou resolution
        SRp_flux_Earth = np.zeros_like(I_ord)
        for iep in range(nep):
            # convolve first to SPIRou resolution
            f_conv = convolve(SRspectra[iep],kernel,normalize_kernel=True,boundary='extend')

            # interpolate to Earth rest frame (for the chosen order)
            star_func = interp1d(SRwl,f_conv,bounds_error=False)
            SRp_flux_Earth[iep] = star_func(WW_star[iep])

        ## normalise the models
        cont_model = []
        for iep in range(len(SRp_flux_Earth)):
            # normalise
            flm = maximum_filter(SRp_flux_Earth[iep],size=100)
            SRp_flux_Earth[iep] /= flm
            cont_model.append(flm)
        O.SR = SRp_flux_Earth

        # Fit model to the data
        ####################################
        ###    STEP 1: normalise data    ###
        ####################################
        fn       = np.zeros((nep,npix))
        cont     = np.zeros_like(fn)
        for iep in range(nep):
            filt      = maximum_filter(I_ord[iep],size=50)
            fn[iep]   = I_ord[iep]/filt
            cont[iep] = filt

        ######################################
        ###    STEP 2: fit scaling param   ###
        ######################################
        scale_params = []
        for iep in range(nep):
            _fm = O.SR[iep]-1
            _fn = fn[iep]-1

            def model(WW_ord,a):
                return a*_fm

            popt,pconv = curve_fit(model,WW_ord,_fn)
            scale_params.append(popt[0])

        ######################################
        ###    STEP 3: apply correction    ###
        ######################################

        I_SR = np.zeros_like(I_ord)
        for iep in range(nep):
            fm_ = O.SR[iep] - 1
            #_fn = 1-fn[iep]
            #I_SR[iep] = ( (_fn/(scale_params[iep]*_fm)) + 1 ) * cont[iep]
            I_SR[iep] = (fn[iep] / (scale_params[iep]*fm_ + 1 ) )

        O.I_raw = I_SR/np.nanmean(I_SR,axis=0)


#### Main reduction
print("START DATA REDUCTION")

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
    if nn==0 and instrument=='igrins':
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
            if sample_residuals: nax = 3
            elif do_hipass: nax = 3
            else: nax = 2
            fig,axes = plt.subplots(nax,1,figsize=(8,8))
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
                    numcall = 15
                    while ((curcall < numcall) and (residrms > maxrms)):
                        #print('On iteration {} of {}'.format(curcall,numcall))
                        #with warnings.simplefilter('ignore',np.RankWarning):
                        z,mask,residrms,test_wlens,normspec = O.fit_blaze(test_wlens, normspec, maxrms,
                                         numcalls=numcall, curcall=curcall,order=15,nbor=15,verbose=False,showplot=False)
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
                O.blaze = blaze

            if not do_hipass and not sample_residuals:
                I_sub1 = I_cl
            elif O.number in [11,19,45,47]:
                I_sub1 = I_cl
                print('skip median out of transit for this order')
            else:
                ### Compute mean spectrum in the stellar rest frame
                V_cl      = c0*(W_cl/O.W_mean-1.)
                I_bary    = move_spec(V_cl,I_cl,V_corr,sig_g)  ## Shift to stellar rest frame
                I_med     = np.median(np.concatenate((I_bary[:n_ini],I_bary[n_end:]),axis=0),axis=0) ## Compute median out-of-transit
                I_med_geo = move_spec(V_cl,np.array([I_med]),-1.*V_corr,sig_g)  ## Move back ref spectrum to Geocentric frame
                I_sub1    = np.zeros(I_cl.shape)

                if doLS:
                    # a stretch/shift of the stellar ref spec to each spectrum (then remove)
                    for kk in range(len(I_cl)):
                        X          = np.array([np.ones(len(I_med_geo[kk])),I_med_geo[kk]],dtype=float).T
                        p,pe       = LS(X,I_sub2[kk])
                        Ip         = np.dot(X,p)
                        I_sub1[kk] = I_sub2[kk]/Ip
                else:
                    I_sub1 = I_cl/I_med_geo
            ### Remove extremities to avoid interpolation errors
            W_sub = W_cl[N_bor:-N_bor]
            I_sub = I_sub1[:,N_bor:-N_bor]#I_cl[:,N_bor:-N_bor]

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
        W_norm1,I_norm1 = O.normalize(W_sub,I_sub,N_med,sig_out,N_bor)
        ### Correct for bad pixels

        W_norm2,I_norm2 = O.filter_pixel(W_norm1,I_norm1,deg_px,sig_out)
        ### END of STEP 2
        print('after normalisation {}'.format(np.isnan(I_norm2.any())))
        #plt.figure()
        #for iep in range(len(I_norm2)):
        #    plt.plot(I_norm2[iep])
        #plt.show()
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

        N_px          = 200
        indw          = np.argmin(np.abs(O.W_fin-O.W_fin.mean()))

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
                O.M_pca = np.exp(x_pca_projected*ist+im) # save that removed for the model reprocessing
                O.ncom_pca = n_com
                # check if within range of DRS dispersion
                disp = np.mean(np.std(O.I_pca[:,indw-N_px:indw+N_px],axis=1))
                drs_disp = 1./O.SNR
                drs_disp_mean = np.mean(drs_disp)
                drs_disp_std = np.std(drs_disp)
                '''while disp>drs_disp_mean+2*drs_disp_std:
                    # if dispersion is greater than 2 errorbars from drs
                    # add a principal component
                    n_com += 1
                    if n_com>10:
                        print('do not use more than 10 PCs')
                        break
                    else:
                        print('adding 1 more PC')
                        pca   = PCA(n_components=n_com)
                        x_pca = np.float32(ff)
                        pca.fit(x_pca)
                        principalComponents = pca.transform(x_pca)
                        x_pca_projected = pca.inverse_transform(principalComponents)
                        O.I_pca = np.exp((ff-x_pca_projected)*ist+im) - 1.0
                        disp = np.mean(np.std(O.I_pca[:,indw-N_px:indw+N_px],axis=1))'''


                NCF[nn] = n_com


                print(n_com,"PCA components discarded")

                if plot:
                    wmin,wmax = O.W_fin.min(),O.W_fin.max()
                    dw = np.average(O.W_fin[1:]-O.W_fin[:-1])
                    extent = (wmin - 0.5 * dw, wmax - 0.5 * dw, nep - 0.5, 0.5)
                    mp2=axes[1].imshow(O.I_pca, extent=extent, interpolation='nearest', aspect='auto',vmin=-0.01,vmax=0.01)
                    fig.colorbar(mp2,ax=axes[1])
                    if not sample_residuals:
                        axes[1].set_xlabel(xlabel)
                        plt.tight_layout()
                        plt.savefig(outroot+"pca_reduced_order{}.png".format(O.number))
                        plt.close()
            # RESIDUAL SAMPLING OF DEEP TELLURIC LINES
            if sample_residuals:
                O.I_pca = O.telluric_residual_sampling(O.W_fin,O.I_pca)
            if do_hipass:
                O.I_pca,mask = O.hipass_filter(O.W_fin,O.I_pca)
                # could apply mask to the data
                # for now dont

            ### ESTIMATES FINAL METRICS
            O.SNR_mes     = 1./np.std(O.I_fin[:,indw-N_px:indw+N_px],axis=1)
            O.SNR_mes_pca = 1./np.std(O.I_pca[:,indw-N_px:indw+N_px],axis=1)

            # mask strong lines
            #if instrument=='igrins' or instrument=='IGRINS':
            #    O.W_fin,O.I_pca =  O.remove_tellurics(dep_min,thres_up)  ### Need a telluric spectrum
            #    if len(O.W_fin)==0:
            #        print("ORDER",O.number,"(",O.W_mean,"nm) discarded (0 pts remaining)")
            #        print("DISCARDED\n")
            #        ind_rem.append(nn)
            #        txt = str(O.number) + "   --\n"
            #        file.write(txt)
            #        continue

            if (plot and sample_residuals) or (plot and do_hipass):
                wmin,wmax = O.W_fin.min(),O.W_fin.max()
                dw = np.average(O.W_fin[1:]-O.W_fin[:-1])
                extent = (wmin - 0.5 * dw, wmax - 0.5 * dw, nep - 0.5, 0.5)
                mp2=axes[2].imshow(O.I_pca, extent=extent, interpolation='nearest', aspect='auto',vmin=-0.01,vmax=0.01)
                fig.colorbar(mp2,ax=axes[2])
                axes[2].set_xlabel(xlabel)
                plt.tight_layout()
                plt.savefig(outroot+"pca_reduced_order{}.png".format(O.number))
                plt.close()

            if plot:
                fig,axes = plt.subplots(2,1,figsize=(8,4))
                wmin,wmax = W_cl.min(),W_cl.max()
                dw = np.average(W_cl[1:]-W_cl[:-1])
                extent = (wmin - 0.5 * dw, wmax - 0.5 * dw, nep - 0.5, 0.5)
                xlabel = 'wavelength (nm)'
                mp1=axes[0].imshow(O.I_pca, extent=extent, interpolation='nearest', aspect='auto',vmin=-0.01,vmax=0.01)
                fig.colorbar(mp1,ax=axes[0])
                axes[0].set_title('Order {}'.format(O.number))


            ### POST-PCA MASKING OF NOISY COLUMNS
            # identify residual tellurics
            nep,npix  = O.I_pca.shape
            std       = np.std(O.I_pca,axis=0)
            thr       = fac*np.nanmedian(std)
            l         = std>thr
            l         = np.tile(l[None,:],(nep,1))
            I_mask    = np.copy(O.I_pca)
            I_mask[l] = np.nan
            # additional masking
            std       = np.nanstd(I_mask)
            mean      = np.nanmean(I_mask)
            l         = np.zeros_like(I_mask,'bool')
            for ipix in range(npix):
                if ((I_mask[:,ipix]-mean)>3.*std).any():
                    l[:,ipix] = True # mask entire column
            #I_mask[np.abs(I_mask)>3.*std] = np.nan
            I_mask[l] = np.nan
            O.I_mask  = I_mask
            mk        = np.isnan(I_mask)
            O.mask    = mk
            O.SNR_mes_mask = 1./np.nanstd(O.I_mask[:,indw-N_px:indw+N_px],axis=1)


            if plot:
                mp2=axes[1].imshow(O.I_mask, extent=extent, interpolation='nearest', aspect='auto',vmin=-0.01,vmax=0.01)
                fig.colorbar(mp2,ax=axes[1])
                axes[1].set_xlabel(xlabel)
                plt.tight_layout()
                plt.savefig(outroot+"masked/masked_order{}.png".format(O.number))
                plt.close()

        txt = str(O.number) + "  " + str(len(O.W_fin)) + "  " + str(np.mean(O.SNR)) + "  " + str(np.mean(O.SNR_mes)) + "  " + str(np.mean(O.SNR_mes_pca)) + "  " + str(n_com) + "\n"
        if inject:
            txt += "injected model found at: {}, Kp: {}, vsys: {}, inj-amp: {}".format(mod_file,inj_Kp,inj_vsys,inj_amp)
        file.write(txt)
        print('\n')

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
MPC = []
NPC = []
for nn in range(len(orders_fin)):
    O  = list_ord_fin[nn]
    WW.append(O.W_fin)
    Ir.append(O.I_pca)
    Imask.append(O.I_mask)
    mask.append(O.mask)
    SNR_mes.append(O.SNR_mes)
    SNR_mes_pca.append(O.SNR_mes_pca)
    MPC.append(O.M_pca)
    NPC.append(O.ncom_pca)
savedata = (orders_fin,WW,Ir,T_obs,phase,window,berv,vstar,airmass,SN,SNR_mes,SNR_mes_pca,Imask,mask,MPC,NPC)
with open(nam_fin, 'wb') as specfile:
    pickle.dump(savedata,specfile)
print("DONE")

t1          = time.time()
print("DURATION:",(t1-t0)/60.,"min")
