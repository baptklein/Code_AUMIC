#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Mar 2022
Edited in Jun 2022

@authors: Baptiste KLEIN, Florian DEBRAS & Annabella MEECH
"""
import numpy as np
import os
from astropy.io import fits
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from astropy.modeling import models, fitting, polynomial
from astropy.stats import sigma_clip
from scipy.optimize import minimize
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import glob
from astropy.time import Time
from scipy.signal import argrelextrema






class Constants:
    def __init__(self):
        self.c0 = 299792.458 #km/s

# -----------------------------------------------------------
# This function reads a time series of PLP IGRINS files
# It returns the spectral matrix of the form (order,epoch,pixel)
# The spectra to read must be stored in a dedicated repository
# -----------------------------------------------------------
def read_igrins_data(repp):
    """
    --> Inputs:     - repp:      Path to the directory containing all the '.fits' files to read

    --> Outputs:    - time_JD: average JD times for AB extracted pairs
                    - wlens: wavelength vectors
                    - data_RAW: PLP data spectral matrix
                    - data_var: PLP data variance matrix
                    - data_sn: PLP data SNR matrix
                    - airms: average airmass value for each extracted AB pair
                    - humidity
    """
    ### Get all data to read
    specfilesH=sorted(glob.glob(repp+'*SDCH*spec.fits'))
    specfilesK=sorted(glob.glob(repp+'*SDCK*spec.fits'))
    varfilesH=sorted(glob.glob(repp+'*SDCH*variance.fits'))
    varfilesK=sorted(glob.glob(repp+'*SDCK*variance.fits'))
    snfilesH=sorted(glob.glob(repp+'*SDCH*sn.fits'))
    snfilesK=sorted(glob.glob(repp+'*SDCK*sn.fits'))
    assert len(specfilesH)==len(varfilesH)==len(specfilesK)==len(varfilesK), "Unequal no. of H and K files"
    print('number of files: {}'.format(len(specfilesH)))

    ### Get shape information for H and K files
    wlfile = fits.open(repp+'SDCH_20210803_0276.wave.fits')
    wlensH = wlfile[0].data
    wlfile = fits.open(repp+'SDCK_20210803_0276.wave.fits')
    wlensK = wlfile[0].data
    wlens  = np.concatenate([wlensK,wlensH]) #descending order
    wlens  = wlens[::-1,:] #ascending order


    ### Initialisation
    ndet, npix = wlens.shape
    nep        = len(specfilesH)-2 # read up to last two files (standard star)
    time_MJD   = np.zeros(nep)
    time_JD    = np.zeros_like(time_MJD)
    airms      = np.zeros_like(time_MJD)
    humidity   = np.zeros_like(time_MJD)
    data_RAW   = np.zeros((ndet,nep,npix))
    data_var   = np.zeros_like(data_RAW)
    data_sn    = np.zeros_like(data_RAW)


    ### open and read files -- arrange data into spectral matrix

    # read up to last two files (standard star observations)
    for ifile in range(len(specfilesH)-2):
        #H-band
        hdu_list = fits.open(specfilesH[ifile])
        image_dataH = hdu_list[0].data
        hdr = hdu_list[0].header
        print('gain: {}'.format(hdr['GAIN']))
        date_begin = hdr['DATE-OBS'] # in UTC
        date_end = hdr['DATE-END']
        t1 = Time(date_begin,format='isot',scale='utc')
        t2 = Time(date_end,format='isot',scale='utc')
        time_JD[ifile] = float(0.5*(t1.jd+t2.jd))
        time_MJD[ifile] = float(0.5*(t1.mjd+t2.mjd))
        airms[ifile] = 0.5*(hdr['AMSTART']+hdr['AMEND']) # average airmass
        humidity[ifile] = hdr['HUMIDITY']
        print(hdr['EXPTIME'])
        print(date_begin, date_end, time_MJD[ifile])

        #variances
        hdu_list   = fits.open(varfilesH[ifile])
        image_varH = hdu_list[0].data

        hdu_list   = fits.open(snfilesH[ifile])
        image_snH  = hdu_list[0].data

        #K-band
        hdu_list = fits.open(specfilesK[ifile])
        image_dataK = hdu_list[0].data
        hdr = hdu_list[0].header
        date_begin = hdr['DATE-OBS'] # in UTC
        date_end = hdr['DATE-END']

        #variances
        hdu_list = fits.open(varfilesK[ifile])
        image_varK = hdu_list[0].data
        hdu_list   = fits.open(snfilesK[ifile])
        image_snK  = hdu_list[0].data

        hdu_list.close()

        #concatenating K and H spectra
        data = np.concatenate([image_dataK,image_dataH])
        var  = np.concatenate([image_varK,image_varH])
        sn   = np.concatenate([image_snK,image_snH])
        data_RAW[:,ifile,:] = data # master matrix
        data_var[:,ifile,:] = var
        data_sn[:,ifile,:]  = sn

    #invert arrays - ascending order in wavelength
    data_RAW = data_RAW[::-1,:,:]
    data_var = data_var[::-1,:,:]
    data_sn  = data_sn[::-1,:,:]
    return time_JD, wlens, data_RAW, data_var, data_sn, airms, humidity

# -----------------------------------------------------------
# This function reads a time series of DRS-provided SPIRou files
# It stores some of the relevant information into "Order" objects
# and returns time series relevant for the analysis
# The spectra to read must be stored in a dedicated repository
# For the time being, the function can only read t.fits extensions
# -----------------------------------------------------------
def read_data_spirou(repp,list_ord,nord):

    """
    --> Inputs:     - repp:      Path to the directory containing all the '.fits' files to read
                                 NOTE: files must be ordered in the chronologic order
                    - list_ord:  List of Order object
                    - nord:      Number of orders -- 49 for SPIRou

    --> Outputs:    - Attributes of Order objects:
                      1. W_raw (Wavelengths vectors)
                      2. I_raw (Time series of spectra)
                      3. blaze (Time series of blaze functions)
                      4. A_raw (Time series of telluric spectra computed from the DRS)
                      5. SNR (list of order S/N values for all observations)
                    - list_ord: upgraded list of orders
                    - airmass: airmass value for each observation
                    - bjd: time vector
                    - snr_mat: 2D matrix containing the S/N value for each observation and order (N_observation,N_order)
    """
    nam_t     = sorted(os.listdir(repp))
    nobs      = len(nam_t)
    airmass   = np.zeros(nobs)
    bjd       = np.zeros(nobs)
    jd        = np.zeros(nobs)
    berv      = np.zeros(nobs)
    snr_mat   = np.zeros((nobs,nord))
    humidity  = np.zeros(nobs)
    for nn in range(nobs):
        nmn          = repp + "/" + str(nam_t[nn])
        hdul_t       = fits.open(nmn)
        airmass[nn]  = float(hdul_t[0].header["AIRMASS"])
        bjd[nn]      = float(hdul_t[1].header["BJD"])
        jd[nn]       = (float(hdul_t[0].header['MJDATE'])+float(hdul_t[0].header['MJDEND']))/2.0+2400000.5

        berv[nn]     = float(hdul_t[1].header["BERV"])
        i            = np.array(hdul_t[1].data,dtype=float) # intensity spectrum
        w            = np.array(hdul_t[2].data,dtype=float) # wavelength vector
        bla          = np.array(hdul_t[3].data,dtype=float) # blaze vector
        atm          = np.array(hdul_t[4].data,dtype=float) # telluric spectrum
        humidity[nn] = float(hdul_t[1].header["RELHUMID"])
        ### Get S/N values
        for mm in range(nord):
            num = 79 - list_ord[mm].number
            if num < 10: key = "EXTSN00" + str(num)
            else: key = "EXTSN0" + str(num)
            sn  = float(hdul_t[1].header[key]) # S/N for each order
            list_ord[mm].SNR.append(sn)
            snr_mat[nn,mm] = sn
        hdul_t.close()
        ## Store Order's attributes
        for mm in range(nord):
            O = list_ord[mm]
            num = 79 - list_ord[mm].number
            O.W_raw.append(w[num])
            O.I_raw.append(i[num])
            O.blaze.append(bla[num])
            O.I_atm.append(atm[num])
    for mm in range(nord):
        O       = list_ord[mm]
        O.SNR   = np.array(O.SNR,dtype=float)
        O.W_raw = np.array(O.W_raw,dtype=float)
        O.I_raw = np.array(O.I_raw,dtype=float)
        O.blaze = np.array(O.blaze,dtype=float)
        O.I_atm = np.array(O.I_atm,dtype=float)
    return list_ord,airmass,jd,berv,snr_mat,humidity

# -----------------------------------------------------------
# Get transit window -- requires batman python module
# Uncomment lines below to use batman module to compute transit flux
# See information in https://lweb.cfa.harvard.edu/~lkreidberg/batman/
# -----------------------------------------------------------
import batman
def compute_transit(Rp,Rs,ip,T0,ap,Porb,ep,wp,limb_dark,uh,T_obs):

    """
    --> Inputs:     - Rp:        Planetary radius
                    - Rs:        Stellar radius (same unit as Rp)
                    - ip:        Transit inclination [deg]
                    - T0:        Mid-transit time (same unit as T_obs -- here: bjd)
                    - ap:        Semi-major-axis [Stellar radius]
                    - Porb:      Planet orbital period (same unit as T_obs)
                    - ep:        Eccentricity of the planet orbit
                    - wp:        Argument of the periapsis for the planet orbit [deg]
                    - limb_dark: Type of limb-darkening model: "linear", "quadratic", "nonlinear" see https://lweb.cfa.harvard.edu/~lkreidberg/batman/
                    - uh:        Limb-darkening coefficients matching the type of model and in the SPIRou band (Typically H or K)
                    - T_obs:     Time vector

    --> Outputs:    - flux:      Relative transit flux (1 outside transit)
    """
#
    params           = batman.TransitParams()
    params.rp        = Rp/Rs
    params.inc       = ip
    params.t0        = T0
    params.a         = ap
    params.per       = Porb
    params.ecc       = ep
    params.w         = wp
    params.limb_dark = limb_dark
    params.u         = uh
    bat              = batman.TransitModel(params,T_obs)
    flux             = bat.light_curve(params)
    return flux



# -----------------------------------------------------------
# Get transit initial and end dates from an input transit curve
# -----------------------------------------------------------
def get_transit_dates(wind):

    """
    --> Inputs:     - flux:  Relative transit flux (e.g., output from batman python module)

    --> Outputs:    - n_ini: Index of the last point before transit
                    - n_end: Index of the first point after transit
    """

    n_ini,n_end = 1,1
    if wind[0] > 0.0: n_ini = 0
    else:
        cc = 0
        while wind[cc] == 0.0:
            cc += 1
        n_ini = cc-1
    if wind[-1] > 0.0: n_end = len(wind)-1
    else:
        cc = n_ini + 1
        while wind[cc] > 0.0:
            cc += 1
        n_end = cc
    return n_ini,n_end


# -----------------------------------------------------------
# Move spectra from one frame to another
# -----------------------------------------------------------
def move_spec(V,I,Vc,sig_g,be=False,fv='extrapolate'):
    """
    --> Inputs:     - V:     Velocity vector (assumed 1D)
                    - I:     Array of flux values (assumed 2D [N_obs,N_wav])
                    - Vc:    Velocimetry correction [km/s]
                    - sig_g: Standard deviation of one pixel
                    - pixel: Binned instrument pixel in wavelength space
                    - kind:  type of interpolatation (scipy interp1D)


    --> Outputs:    - I_al:  2D matrix of Vc-corrected spectra

    """

    I_al    = np.zeros((len(Vc),len(V)))
    dddv    = np.linspace(-3.*sig_g,3.*sig_g,30)
    G       = normal_law(dddv,0.0,sig_g)
    step    = dddv[1]-dddv[0]

    for ii in range(len(Vc)):

        ### Depending on which frame we're moving into
        if len(I) == len(Vc): fi = interp1d(V,I[ii],kind="cubic",bounds_error=be,fill_value=fv)#fill_value="extrapolate")
        else:                 fi = interp1d(V,I[0],kind="cubic",bounds_error=be,fill_value=fv)#fill_value="extrapolate")

        I_tmp     = step * (fi(V+Vc[ii]+dddv[0])*G[0]+fi(V+Vc[ii]+dddv[-1])*G[-1]) * 0.5
        for hh in range(1,len(dddv)-1):
            I_tmp += step*fi(V+Vc[ii]+dddv[hh])*G[hh]
        I_al[ii] = I_tmp
    return I_al







# -----------------------------------------------------------
# Compute RV signature induced by the planet on its host star
# Assuming circular orbit for the planet
# -----------------------------------------------------------
def get_rvs(t,k,p,t0):

    """
    --> Inputs:     - t:   Time vector
                    - k:   Semi-amplitude of the planet-induced RV signal on the host star
                    - p:   Planet orbital period
                    - t0:  Planet mid-transit time

    --> Outputs:    - Planet-induced RV signature for the input time values
    """

    return  k*np.sin(2.*np.pi/p * (t0-t))

# -----------------------------------------------------------
# Compute planet RV in the stellar rest frame
# -----------------------------------------------------------
def rvp(phase,k,v):
    """
    --> Inputs:     - phase: Planet orbital phase (T-T_obs)/Porb
                    - k:     Semi-amplitude of the planet RV
                    - v:     Planet RV at mid-transit

    --> Outputs:    - Planet RV for the input orbital phases
    """
    return k*np.sin(2.*np.pi*phase)+v


# -----------------------------------------------------------
# Simple hyperbola
# -----------------------------------------------------------
def hyp(par,xx):
    return par[0]/xx + par[1]

# -----------------------------------------------------------
# Simple inverse hyperbola
# -----------------------------------------------------------
def hyp_inv(par,yy):
    return par[0]/(yy-par[1])

# -----------------------------------------------------------
# Return least-square difference between a hyperbola for 'par'
# parameters and data yy.
# xx is the X-axis vector
# -----------------------------------------------------------
def crit_hyp(par,xx,yy):
    y_pred = hyp(par,xx)
    return np.sum((yy-y_pred)**(2))

def poly(par,xx):
    return par[0] + par[1]*xx + par[2]*xx**2

def poly_inv(par,yy):
    c,b,a = par
    ci = b**2/(4*a)
    cj = b/(2*a)
    tfr = yy + ci - c
    return np.sqrt(tfr/a) - cj

def crit_poly(par,xx,yy):
    y_pred = poly(par,xx)
    return np.sum((yy-y_pred)**2)


# -----------------------------------------------------------
# Compute Order to mean wavelength equivalence
# Usage: Plot order number as X-axis and mean wavelengths as Y axis
# In practice: fits an hyperbola between order nb and mean wavelength
# See function plots.plot_orders for more information
# -----------------------------------------------------------
def fit_order_wave(LO,wm_fin,instrument):

    """
    --> Inputs:     - LO: list of order numbers
                    - wm_fin: list of the mean wavelengths corresponding to LO
                    - instrument: igrins or spirou

    --> Outputs:    - WW: Wavelength ticks for the plot
                    - LO_pred: order numbers corresponding to WW
                    - LO_predt: densely-sampled list of orders for minor ticks locators
    """

    if instrument == 'IGRINS' or instrument == 'igrins':
        LO_tot  = np.arange(1,54)
        WWT      = np.linspace(1400,2500,11)
        WW       = np.array([1400,1650,1900,2150,2400,2500],dtype=int)
        par0    = np.array([1.,1.,1.],dtype=float)
        res     = minimize(crit_poly,par0,args=(LO,wm_fin))
        p_best  = res.x
        pp       = poly(p_best,LO_tot)
        LO_predt = poly_inv(p_best,WWT)
        LO_pred  = poly_inv(p_best,WW)
    elif instrument == 'SPIROU' or instrument == 'spirou':
        LO_tot  = np.arange(29,81)
        WWT      = np.linspace(2400,900,16)
        WW       = np.array([2400.0,2100,1800,1500,1200,1000],dtype=int)
        par0    = np.array([100,1400.0],dtype=float)
        res     = minimize(crit_hyp,par0,args=(LO,wm_fin))
        p_best  = res.x
        pp       = hyp(p_best,LO_tot)
        LO_predt = hyp_inv(p_best,WWT)
        LO_pred  = hyp_inv(p_best,WW)
    else:
        sys.exit('Instrument option not compatible. Choose SPIROU or IGRINS.')

    return WW,LO_pred,LO_predt


# -----------------------------------------------------------
# Simple least-squares estimate
# Same as using numpy.linalg.lstsq (https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
# -----------------------------------------------------------
def LS(X,Y,Si=[]):

    if len(Si) > 0:
        A    = np.dot(np.dot(X.T,Si),X)
        b    = np.dot(np.dot(X.T,Si),Y)
    else:
        A    = np.dot(X.T,X)
        b    = np.dot(X.T,Y)
    Ainv = np.linalg.inv(A)
    par  = np.dot(Ainv,b)
    return par,np.sqrt(np.diag(Ainv))

# -----------------------------------------------------------
# Iterative polynomial fit with outlier removal
# Documentation: https://docs.astropy.org/en/stable/api/astropy.modeling.fitting.FittingWithOutlierRemoval.html
# -----------------------------------------------------------
def poly_fit(x,y,deg,sig,n_iter=3):

    """
    --> Inputs:     - x:      X vector
                    - y:      Data to fit
                    - deg:    Degree of the polynomial to fit
                    - sig:    Threshold for the outlier removal [in sigma]
                    - n_iter: Number of iterations for the fit

    --> Outputs:    - fitted_model (function): best-fitting model after last iteration
                    - filtered_data: data after outlier removal
    """

    pol_f     = polynomial.Polynomial1D(deg) ### Init polynom
    fit       = fitting.LinearLSQFitter()   ### Init optim method
    or_fit    = fitting.FittingWithOutlierRemoval(fit,sigma_clip,niter=n_iter, sigma=sig)  ### Do the fit at sig sigma level
    or_fitted_model,mask = or_fit(pol_f,x,y)
    filtered_data        = np.ma.masked_array(y,mask=mask)
    fitted_model         = fit(pol_f,x,filtered_data)
    return fitted_model,filtered_data



def normal_law(v,mu,sigma):
    g = 1./(np.sqrt(2.*np.pi)*sigma) * np.exp(-0.5*((v-mu)/(sigma))**(2))
    return g


def stretch_shift(x,cs_data,aa,bb):
    xx = x*aa + bb
    data_int = interpolate.splev(xx,cs_data,der=0)
    return data_int


# -----------------------------------------------------------
# Apply PCA to an input sequence of spectra:
# 1. Apply PCA to the sequence (centered + reduced)
# 2. Set the first N_comp_pca components to 0 (i.e. components associated to largest variance contribution in the data)
# 3. Project back to the input 'data frame'
# -----------------------------------------------------------
def make_pca(I,N_comp_pca,return_all=False):

    """
    --> Inputs:     - I:          Input sequence of spectra (2D matrix)
                    - N_comp_pca: Number of PCA components to remove
                    - return_all: Boolean --> if True, return discared components back into the data frame

    --> Outputs:    - e_var: Relative contribution of each component to the variance in the data
                    - I_pca: Sequence of spectra after removing the first N_comp_pca
                    - I_del: Removed components (same shape as I) -- NOTE: only if return_all == True

    """

    ### Preprocessing: center + reduce input matrix
    I2       = np.copy(I)
    Im       = I2.mean(axis = 0)
    Is       = I2.std(axis = 0)

    ### PCA decomposition
    U, s, VT = np.linalg.svd((I2-Im)/Is)
    comp     = VT[:len(I2)]
    comp2    = np.copy(comp)
    var      = s ** 2 / I2.shape[0]
    e_var    = var/var.sum()
    X_new    = s * U

    ### Set the first components to 0
    comp2[:N_comp_pca]  = np.zeros((N_comp_pca,len(comp[0])))

    ### Project back into initial basis
    I_pca               = Is*np.dot(X_new,comp2) + Im

    if return_all: # compute individually each removed components
        I_del  = np.zeros((N_comp_pca,len(I),len(I[0])))
        for ii in range(N_comp_pca):
            cc        = np.zeros(I.shape)
            cc[ii]    = comp[ii]
            I_del[ii] = np.dot(X_new,cc)
        return e_var,I_pca,I_del
    else: return e_var,I_pca


# -----------------------------------------------------------
# Compare the dispersion at the center of each spectrum (in
# each order) to the photon noise provided by the SPIRou DRS
# -----------------------------------------------------------
def plot_spectrum_dispersion(lord,nam_fig,instrument):

    """
    --> Inputs:     - lord: list of Order objects
                    - nam_fig: filename
                    - instrument: igrins or spirou

    --> Outputs:    - Plot displayed

     """

    # Initialization
    rms_sp     = np.zeros(len(lord))
    rms_sp_s   = np.zeros(len(lord))
    rms_drs    = np.zeros(len(lord))
    rms_drs_s  = np.zeros(len(lord))
    rms_pca    = np.zeros(len(lord))
    rms_pca_s  = np.zeros(len(lord))
    rms_mask   = np.zeros(len(lord))
    rms_mask_s = np.zeros(len(lord))
    wmean      = np.zeros(len(lord))
    LO         = np.zeros(len(lord),dtype=int)

    for kk in range(len(lord)):
        O              = lord[kk]
        disp_mes       = 1./O.SNR_mes
        disp_drs       = 1./O.SNR
        disp_pca       = 1./O.SNR_mes_pca
        disp_mask      = 1./O.SNR_mes_mask
        rms_sp[kk]     = np.mean(disp_mes)
        rms_sp_s[kk]   = np.std(disp_mes)
        rms_drs[kk]    = np.mean(disp_drs)
        rms_drs_s[kk]  = np.std(disp_drs)
        rms_pca[kk]    = np.mean(disp_pca)
        rms_pca_s[kk]  = np.std(disp_pca)
        rms_mask[kk]   = np.mean(disp_mask)
        rms_mask_s[kk] = np.std(disp_mask)
        wmean[kk]      = O.W_mean
        LO[kk]         = O.number

    # Compute wavelength-order number correspondance
    WW,LO_pred,LO_predt = fit_order_wave(LO,wmean,instrument)
    plt.figure(figsize=(12,5))
    ax = plt.subplot(111)
    ax.errorbar(LO,rms_sp,rms_sp_s,fmt="*",color="k",label="Reduced data",capsize=10.0,ms=10.)
    ax.errorbar(LO,rms_pca,rms_pca_s,fmt="^",color="g",label="After PCA",capsize=10.0,ms=7.5)
    ax.errorbar(LO,rms_drs,rms_drs_s,fmt="o",color="m",label="DRS",capsize=8.0)
    ax.errorbar(LO,rms_mask,rms_mask_s,fmt="+",color="skyblue",label="Masked post PCA",capsize=10.0,ms=10.)

    ax.legend(ncol=2)
    ax2 = ax.twiny()
    ax2.set_xticks(LO_pred)
    ax2.set_xlabel("Wavelength [nm]")
    ax2.set_xticklabels(WW)
    ax2.xaxis.set_minor_locator(ticker.FixedLocator(LO_predt))
    #ax.set_xlim(30,80)
    #ax2.set_xlim(30,80)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel("Spectrum dispersion")
    ax.set_xlabel("Order number")
    ax.set_yscale("log")
    plt.subplots_adjust(wspace=0.5,hspace = 0.)
    plt.savefig(nam_fig,bbox_inches="tight")
    plt.close()



#### Main class -- Order

class Order:


    def __init__(self,numb):

        ### Generic information
        self.number       = numb    # Order number (in absolute unit -- 79: bluest; 31: reddest)
        self.W_mean       = 0.0     # Mean order wavelength
        self.SNR          = []      # DRS-computed S/N at the center of the order
        self.SNR_mes      = []      # Empirical estimate of the SNR before PCA
        self.SNR_mes_pca  = []      # Empirical estimate of the SNR after PCA

        ### Raw data information
        self.W_raw    = []      # Wavelength vectors for the raw observations - 2D matrix (time-wavelength)
        self.I_raw    = []      # Time series of observed spectra from the SPIRou DRS - 2D matrix (time-wavelength)
        self.I_atm    = []      # Time series of Earth atmosphere spectra computed from the observations using Artigau+2014 method - DRS-provided 2D matrix (time-wavelength)
        self.blaze    = []      # Time series of blaze functions - 2D matrix (time-wavelength)

        ### Data reduction information
        self.W_fin    = []      # Final wavelength vector in the Geocentric frame
        self.W_bary   = []      # Final wavelength vector in the stellar rest frame
        self.I_fin    = []      # Reduced flux matrix before PCA cleaning
        self.I_pca    = []      # Reduced flux matrix after PCA cleaning

        ### Model and correlation parameters
        self.Wm       = []
        self.Im       = []
        self.corr     = []
        self.M_pca    = []
        self.ncom_pca = []
        self.std_fit  = []



    # -----------------------------------------------------------
    # Pre-process of the DRS-provided spectra:
    # 1. Blaze normalization process
    # 2. Remove NaNs from each spectrum and convert sequences of
    #    spectra into np.array square matrices
    # -----------------------------------------------------------
    def remove_nan(self):

        """
        --> Inputs:      - Order object

        --> Outputs:     - Boolean: 1 --> order empty as NaNs everywhere; 0 otherwise
        """

        ### Remove blaze
        I_bl = self.I_raw/self.blaze


        ### Spot the NaNs:
        ### In "*t.fits" files, regions of high telluric absorptions are replaced by NaNs
        ### as no precise estimation of the flux could be carried out
        ### Here we build a vector 'ind' stroring the position of the NaNs in every spectrum
        ind   = []
        for nn in range(len(I_bl)):
            i = np.where(np.isfinite(I_bl[nn])==True)[0]
            ind.append(i)
        r  = np.array(list(set.intersection(*map(set,ind))),dtype=int)
        r  = np.sort(np.unique(r))

        ### remove the NaNs
        I_ini = []
        W_ini = []
        B_ini = []
        A_ini = []
        for nn in range(len(I_bl)):
            I_ini.append(I_bl[nn,r])
            W_ini.append(self.W_raw[nn,r])
            A_ini.append(self.I_atm[nn,r])
            B_ini.append(self.blaze[nn,r])

        ### Convert into 2D array object
        self.I_raw  = np.array(I_ini,dtype=float)
        self.W_raw  = np.array(W_ini,dtype=float)[0]
        self.I_atm  = np.array(A_ini,dtype=float)
        self.B_raw  = np.array(B_ini,dtype=float)
        self.W_mean = self.W_raw.mean()   ### Compute mean of the actual observations

        ### Remove the order if it contains only NaNs
        if len(self.I_raw[0]) == 0:
            tx = "\nOrder " + str(self.number) + " is empty and thus removed from the analysis"
            print(tx)
            return 1
        else:
            return 0









    # -----------------------------------------------------------
    # Remove regions of strong telluric absorption
    # 1. From DRS-provided telluric spectrum, spot all regions
    #    with more than 'dep_min' relative absorption depth.
    # 2. For all regions identified, remove points on both sides
    #    until reaching thres_up relative absorption level wrt
    #    the continuum
    # -----------------------------------------------------------
    def remove_tellurics(self,dep_min,thres_up):

        """
        --> Inputs:     - Order object
                        - dep_min: Threshold (in relative absorption unit) above
                                   which lines are removed
                        - thres_up: When removing a region of strong absorption,
                                    all points are remove until reaching a relative
                                    absorption of 'thres_up'

        --> Outputs:    - self.I_cl; self.W_cl
        """

        ### Identify regions of strong telluric absorption from the median DRS-provided
        ### telluric spectrum
        Am      = np.median(self.I_atm,axis=0)
        ind_A   = np.where(Am<1-dep_min)[0]

        ### Identify regions adjacent to the telluric absorption lines spotted in the previous step
        ind_tel = []
        for ii in range(len(ind_A)):
            i0     = ind_A[ii]
            n_ini2 = 0
            while Am[i0-n_ini2] < 1 - thres_up:
                if i0 - n_ini2 == 0: break
                n_ini2 += 1
            n_end2 = 0
            while Am[i0+n_end2] < 1 - thres_up:
                if i0 + n_end2 == len(Am)-1: break
                n_end2 += 1
            itel = np.arange(i0-n_ini2,i0+n_end2+1)
            ind_tel.append(itel)
        if len(ind_tel)>0: ind_tel = np.sort(np.unique(np.concatenate(ind_tel)))
        else:              ind_tel = []

        ### Remove regions of strong telluric absorption
        I_cl    = np.delete(self.I_raw,ind_tel,axis=1)
        W_cl    = np.delete(self.W_raw,ind_tel)
        return W_cl,I_cl


    # -----------------------------------------------------------
    # Fit a blaze function (the continuum) to the (IGRINS) spectra
    #
    # -----------------------------------------------------------
    def fit_blaze(self, Ws, Is, rms_thres, numcalls=10, curcall=0, order=5, nbor=50,
                 verbose=False, showplot=False):
        """
        --> Inputs:     - Order object
                        - Ws:        Wavelength vector
                        - Is:        2D median-normalised flux matrix
                        - rms_thres: threhsold rms for the fit in normalised rms
                                     e.g a threshold of 0.01 will keep iterating
                                     until the rms of the residuals is 1%
                        - numcalls:  the max number of iterations
                        - curcall:   store current iteration
                        - order:     order of the polynomial to fit
                        - verbose:   print info
                        - showplot:  plot things

        --> Outputs:    - z:         fitted Polynomial
                        - mask:      mask applied to the spectra
                        - residrms:  rms of the residuals
                        - Ws:        newly masked wavelength vector
                        - Is:        masked flux matrix
        """
        #get wavelength range:
        rwav = max(Ws) - min(Ws)

        #center wavelength range about zero:
        wavcent = Ws - min(Ws) - rwav/2.

        #normalize the spectrum:
        #normspec = spec/max(spec)

        #fit a polynomial to the data:
        z = np.polyfit(Ws, Is, order)

        #make a function based on those polynomial coefficients:
        cfit = np.poly1d(z)

        #make a lower threshold that is offset below the continuum fit. All points
        #below this fit (i.e. spectral lines) will be excluded from the fit in the
        #next iteration.
        thresh = cfit(Ws) - (0.5 * (1. / (curcall + 1)))
        mask = np.where(Is > thresh)[0]
        #print(mask)
        a = np.ones(len(Is),'bool')
        a[mask]  = 0
        a[:nbor] = 0# exclude the edges from the mask though
        a[-nbor:] = 0
        if showplot:
            #plot the original spectrum:
            plt.plot(Ws, Is)
            #overplot the continuum fit
            plt.plot(Ws, cfit(Ws))
            plt.plot(Ws, thresh)
            plt.plot(Ws[a], Is[a], 'r.')
            plt.show()

        mask = np.invert(a)

        residrms = np.std(Is/cfit(Ws))
        if (verbose is True):
            print('now in iteration {0}'.format(curcall))
            print('residrms is now {0:.5f}'.format(residrms))
            print('maxrms is {0})'.format(rms_thres))
            #print('z is: {}'.format(z))

        #now un-center the wavelength range:
        #if curcall == 0:
            #z[-1] = z[-1] - min(wav) - wavspread/2.

        Ws = Ws[mask]
        Is = Is[mask]
        return z,mask,residrms,Ws,Is
    # -----------------------------------------------------------
    # Normalize and remove outlier for each residual spectrum
    # -- Step 1.4 of the data reduction procedure
    # Iterative process: For each spectrum I
    # 1. Apply median filter and normalize I by the smoothed spectrum
    # 2. Remove outliers (+a few adjacent points)
    # 3. Repeat the process until no outlier identified
    # 4. Interpolate the spectrum after outlier removal and replace
    #    location of former outliers by averaged of closest non-NaN points
    # 5. Re-apply median filter and normalize the spectrum
    # -----------------------------------------------------------
    def normalize(self,Ws,Is,N_med,sig_out,nbor=30):

        """
        --> Inputs:     - Order object
                        - Ws:      Wavelength vector
                        - Is:      2D median-normalised flux matrix
                        - N_med:   Cut-off frequency of the median filter (nb of points of the sliding window of
                                   the moving median)
                        - sig_out: Threshold for outlier removal (in sigma)
                        - N_adj:   Number of adjacent points to each outlier removed with the outlier
                        - nbor:    Number of points removed at each edge of the order

        --> Outputs:    - self.I_norm, self. W_norm
        """

        ### Initialization
        ind_out_fin = []
        #I_norm      = np.zeros((len(Is),len(Ws)-int(2*nbor)))
        I_norm      = []
        I_corr      = np.zeros(Is.shape)
        I_med_tot   = np.zeros(Is.shape)
        ind_fin     = []

        for ii in range(len(Is)):
            I           = np.copy(Is[ii]) # Temp. spectrum
            W           = np.copy(Ws)

            Im   = median_filter(I,N_med)
            In   = I/Im
            filt = sigma_clip(In,sigma=sig_out,maxiters=5,masked=True)
            ind2 = np.where(filt.mask)[0]
            ind_fin.append(ind2)

        ind_fin = np.sort(np.unique(np.concatenate(ind_fin)))
        Is2     = np.delete(Is,ind_fin,axis=1)
        Ws2     = np.delete(Ws,ind_fin)
        I_norm  = []
        for nn in range(len(Is2)):
            Im   = median_filter(Is2[nn],N_med)
            In   = Is2[nn]/Im
            I_norm.append(In)
        I_norm = np.array(I_norm,dtype=float)[:,nbor:-nbor]
        W_norm = Ws2[nbor:-nbor]
        return W_norm,I_norm





    # -----------------------------------------------------------
    # Detrending with airmass
    # -- Step 1.5 of the data reduction procedure
    #  Fit a polynomial model of airmass to the sequence of normalized
    #  spectra
    # -----------------------------------------------------------
    def detrend_airmass(self,W,I,airmass,deg=2):

        """
        --> Inputs:     - Order object
                        - airmass: vector of airmass
                        - deg:     Degree of the airmass model (2 in generally sufficient)


        --> Outputs:    - self.I_det
                        - pb: vector of best-fitting spectra
                        - I_pred: Best-fitting modelled sequence of spectra
        """
        indw    = np.argmin(np.abs(W-self.W_mean))
        #print(indw)
        #print(self.W_mean)
        if indw<=200:
            # then beginning of order has been chopped
            indw += 200 # get as close as poss to centre
        elif indw>=len(W)-200:
            # end of order has been chopped
            indw -= 200

        #print(np.nanstd(I[:,indw-200:indw+200],axis=1))
        COV_inv = np.diag(1./np.nanstd(I[:,indw-200:indw+200],axis=1)**(2)) ## Assumes that normalized spectra dominated by white noise
        X       = np.ones((len(I),deg+1))
        for ii in range(deg): X[:,ii+1] = airmass**(ii+1)
        pb,pbe = LS(X,I,COV_inv)
        I_pred = np.dot(X,pb)
        I_det  = I - I_pred
        return I_det




    # -----------------------------------------------------------
    # Filter bad pixels
    # -- Step 1.6 of the data reduction procedure
    #  1. Compute standard deviation for each pixel
    #  2. Fit a parabola to the distribution of pixel dispersion
    #  3. Remove all outliers above fit
    # -----------------------------------------------------------
    def filter_pixel(self,W,I,deg_px=2,sig_px=4.):

        """
        --> Inputs:     - Order object
                        - W:       Wavelength vector
                        - I:       Intensity values
                        - deg_px:  degree of the polynomial fit (2 is generally sufficient)
                        - sig_px:  Threshold for outlier removal

        --> Outputs:    - self.I_red, self.W_red
        """

        n_iter_fit    = 10 ### Number of iterations for the iterative polynomial fit with outliers removal
                           ### See documentation on astropy --> https://docs.astropy.org/en/stable/api/astropy.modeling.fitting.FittingWithOutlierRemoval.html
        rms_px        = I.std(axis=0) # Dispersion of each pixel (computed along the time axis)
        WW            = W - np.mean(W) # Wavelength vector for the fit
        model,filt_px = poly_fit(WW,rms_px,deg_px,sig_px,n_iter_fit) ### See functions below
        rms_pred      = model(WW) # Best prediction

        ### Identify and remove outliers
        ind_px        = []
        for ii in range(len(filt_px)):
            if filt_px[ii] != "--": ind_px.append(ii)
            elif rms_px[ii] < rms_pred[ii]: ind_px.append(ii)
        return W[ind_px],I[:,ind_px]




    def tune_pca(self,Nmap=5,thr=1.0):
        """
        --> Inputs:     - Order object
                        - Nmap:    Number of white noise map used to compute the threshold
        --> Outputs:    - ncf:     Number of PCA components to remove
        """
        N_px          = 200    ### Half nb of px used to compute the dispersion for each pixel
        n_iter_fit    = 10     ### Number of iterations for the polynomial fit to the px std
        ### Initialisation:
        fff           = self.I_fin
        fm            = np.tile(np.nanmean(fff,axis=0),(len(fff),1))
        fs            = np.tile(np.nanstd(fff,axis=0),(len(fff),1))

        fff           = (fff-fm)/fs
        ### Determinate S/N at the center of the order for each epoch
        indw          = np.argmin(np.abs(self.W_fin-self.W_fin.mean()))
        if indw<=N_px:
            indw+=N_px
        elif indw>=len(self.W_fin)-N_px:
            indw-=N_px
        std_mes       = np.std(fff[:,indw-N_px:indw+N_px],axis=1)
        ### Determine the blaze amplification function (border of the order)
        WW            = self.W_fin - self.W_mean
        std_px        = np.std(fff,axis=0)
        std_in        = np.dot(std_mes.reshape((len(fff),1)),np.ones((1,len(WW))))
        model,filt    = poly_fit(WW,std_px,2,5,n_iter_fit)
        ampl          = model(WW)/np.min(model(WW))
        ### Generate noise maps, amplify them, and apply PCA
        thres         = np.zeros(Nmap) ### Store highest eigenvalue for each noise map
        for ii in range(Nmap):
            ### Generate noise map
            NN    = np.random.normal(0.0,std_in*ampl)
            Nm    = np.tile(np.nanmean(NN,axis=0),(len(NN),1))
            Ns    = np.tile(np.nanstd(NN,axis=0),(len(NN),1))
            Nf    = (NN-Nm)/Ns
            #print()
            ### Apply PCA
            pca   = PCA(n_components=len(Nf))
            pca.fit(np.float32(Nf))
            var       = pca.explained_variance_ratio_
            ###Store highest eigenvalue
            thres[ii] = np.max(var)
        ### Apply PCA to observed data
        pca   = PCA(n_components=len(NN))
        x_pca = np.float32(fff)
        pca.fit(x_pca)
        var   = pca.explained_variance_ratio_
        #plt.plot(var,"*")
        #plt.axhline(thr*np.max(thres))
        #plt.show()
        ### Nb of components: larger than 2*max highenvalue
        ncf   = len(np.where(var>thr*np.max(thres))[0])
        return ncf,ampl

    def telluric_residual_sampling(self,W,I,wlcen=None):
        """
        sample the residuals of strong telluric lines post PCA, fit function
        and apply to entire spectral matrix
        --> Inputs:     - Order object
                        - W:       Wavelength vector
                        - I:       PCA-processed spectra
                        - wlcen:   Optionally manually input wavelengths of the lines
                                    to sample, if not O.I_atm is used

        --> Outputs:    - self.I_red, self.W_red
        """
        nep,npix = I.shape
        spec     = np.copy(I)
        if wlcen is None:
            print('no wavelengths supplied, identifying deep telluric wavelengths to sample')
            # identify strong telluric line wavelengths using skycalc Models
            tell_model = np.copy(self.I_atm)
            tell_wlens = np.copy(self.W_atm[0]) # should be same in each ep
            tell_model = np.median(tell_model,axis=0) # take average over time
            minima_idx = argrelextrema(tell_model,np.less)[0] # find all minima
            a          = (tell_model[minima_idx]<0.7)*(tell_model[minima_idx]>0.1)
            select     = minima_idx[a] # filter deepest lines
            if len(select)<2:
                a      = (tell_model[minima_idx]<0.95)*(tell_model[minima_idx]>0.1)
                select = minima_idx[a]
            wlcen      = tell_wlens[select]

        # Building sampling vector
        smpl = np.zeros(nep)
        for wl in wlcen:
            dwl = np.abs(wl-W)
            iline = dwl < 0.15 # indices within proximity of line
            for iep in range(nep):
                arr = spec[iep].copy()
                smpl[iep] += arr[iline].sum()
        # De-trend all the data columns
        # check strong tellurics found
        if (smpl==0.).all():
            # don't perform residual sampling
            print('no strong lines sampled')
        else:
            spec += 1.0
            for ipix in range(npix):
                # fit polynomial in each pixel
                cf = np.polyfit(smpl,spec[:,ipix],2)
    #           fit = cf[0]*smpl + cf[1]
                fit = cf[0]*smpl**2 + cf[1]*smpl + cf[2]
                spec[:,ipix] /= fit
            spec -= 1.0
        return spec,wlcen

    def hipass_filter(self,W,I):
        spec = I.copy() + 1
        nep,npix = spec.shape
        mask = np.zeros_like(spec,'bool')
        l = np.isfinite(spec) == False
        mask[l] = 1
        thre = 2.0
        rms = np.std(spec, axis=0)
        xx = np.arange(npix)
    #    yy = np.arange(nf)
        # Clipping data at 'thre' times the r.m.s.
        ino = rms > thre*np.nanmedian(rms)
        mask[:,ino] = 1
        # Do filtering in spectral direction
        for iep in range(nep):
            arr = spec[iep,].copy()
            mask_arr = mask[iep,]
            iok = (mask_arr == 0) * np.isfinite(arr)
            cf = np.polyfit(xx[iok],arr[iok],2)
            fit = cf[0]*xx**2 + cf[1]*xx + cf[2]
            spec[iep,] /= fit

        spec -= 1.0
        return spec, mask
