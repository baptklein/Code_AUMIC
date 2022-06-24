#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/2022 12:54:11 2021

@author: Baptiste & Florian
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pickle
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.io import fits
from functions import *
import glob


# define planet b parameters
# from Martioli+2021
P        = 8.463000 # period in days
t0       = 2458330.39051 # known mid-transit time in JD
dir_data = "Data/IGRINS/"   #Sep18 #Jun19


### Transit parameters -- Compute the transit window
### Using batman python package https://lweb.cfa.harvard.edu/~lkreidberg/batman/
### Get the limb-darkening coefficients in H band from Claret+2011: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/529/A75
Rp       = 25929.97  #Planet radius  [km]
Rs       = 522255.0  #Stellar radius [km]
ip       = 89.5      #Transit incl.  [deg]
ap       = 18.476     #Semi-maj axis  [R_star]
ep       = 0.0     #Eccentricity of Pl. orbit
wp       = 90.0     #Arg of periaps [deg]
ld_mod   = "nonlinear"     #Limb-darkening model ["nonlinear", "quadratic", "linear"]
ld_coef  = [1.2783,-1.5039,1.2042,-0.3833] # Claret et al. 2011


### Stellar radial velocity info
Ks        = 0.006    #RV semi-amplitude of the star orbital motion due to planet [km/s]
V0        = -4.71    #Stellar systemic velocity [km/s]





### Name of the pickle file to store the info in
outroot = "Input_data/"
if not os.path.exists(outroot):
    os.makedirs(outroot)
name_fin = outroot+"data_igrins.pkl"

### Get all data to read
specfilesH=sorted(glob.glob(dir_data+'*SDCH*spec.fits'))
specfilesK=sorted(glob.glob(dir_data+'*SDCK*spec.fits'))
varfilesH=sorted(glob.glob(dir_data+'*SDCH*variance.fits'))
varfilesK=sorted(glob.glob(dir_data+'*SDCK*variance.fits'))
snfilesH=sorted(glob.glob(dir_data+'*SDCH*sn.fits'))
snfilesK=sorted(glob.glob(dir_data+'*SDCK*sn.fits'))
if not len(specfilesH)==len(varfilesH)==len(specfilesK)==len(varfilesK): print(False)
print('number of files: {}'.format(len(specfilesH)))


### Get shape information for H and K files
wlfile = fits.open(dir_data+'SDCH_20210803_0276.wave.fits')
wlensH = wlfile[0].data
wlfile = fits.open(dir_data+'SDCK_20210803_0276.wave.fits')
wlensK = wlfile[0].data
wlens = np.concatenate([wlensH,wlensK])


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
    date_begin = hdr['DATE-OBS'] # in UTC
    date_end = hdr['DATE-END']
    t1 = Time(date_begin,format='isot',scale='utc')
    t2 = Time(date_end,format='isot',scale='utc')
    time_JD[ifile] = float(0.5*(t1.jd+t2.jd))
    time_MJD[ifile] = float(0.5*(t1.mjd+t2.mjd))
    airms[ifile] = 0.5*(hdr['AMSTART']+hdr['AMEND']) # average airmass
    humidity[ifile] = hdr['HUMIDITY']
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

    #concatatingin K and H spectra
    data = np.concatenate([image_dataH,image_dataK])
    var  = np.concatenate([image_varH,image_varK])
    sn   = np.concatenate([image_snH,image_snK])
    data_RAW[:,ifile,:] = data # master matrix
    data_var[:,ifile,:] = var
    data_sn[:,ifile,:]  = sn

plt.plot(time_MJD,"+")
plt.show()






tmid = t0
i = 0
while tmid<time_JD[0]:
    tmid += P
    i += 1
print('')
print('mid-transit time: {}'.format(tmid))


### Compute transit window
phase        = (time_JD-tmid)/P
t0           = tmid
flux         = compute_transit(Rp,Rs,ip,t0,ap,P,ep,wp,ld_mod,ld_coef,time_JD)
window       = (1-flux)/np.max(1-flux)
print("DONE")




ind_sort = np.argsort(wlens.mean(axis=1))
wlens    = wlens[ind_sort]
data_RAW = data_RAW[ind_sort]
dnorm    = np.nanmax(data_RAW,axis=2)
data_fin = np.zeros_like(data_RAW)
for zz in range(len(data_RAW)):
    for yy in range(len(data_RAW[0])):
        data_fin[zz,yy] = data_RAW[zz,yy]/dnorm[zz,yy]
data_RAW = data_fin
data_var = data_var[ind_sort]
data_sn  = data_sn[ind_sort]
SN       = np.nanmean(data_sn,axis=2)
nord     = len(SN)




### Remove NaNs
I_corr,W_corr,var_corr = [],[],[]
for zz in range(nord):

    I_bl  = data_RAW[zz]
    W     = wlens[zz]

    ind   = []
    for nn in range(len(I_bl)):
        i = np.where(np.isfinite(I_bl[nn])==True)[0]
        ind.append(i)
    r  = np.array(list(set.intersection(*map(set,ind))),dtype=int)
    r  = np.sort(np.unique(r))

    ### remove the NaNs
    I_ini = []
    W_ini = W[r]
    for nn in range(len(I_bl)):
        I_ini.append(I_bl[nn,r])

    print("Order",zz,":",len(r))

    I_corr.append(np.array(I_ini,dtype=float))
    W_corr.append(np.array(W_ini,dtype=float))



c0 = 299792.458
for nn in range(len(W_corr)):
    WW   = W_corr[nn]
    WM   = np.mean(WW)
    diff = np.diff(WW)
    print(np.mean(diff)*c0/WM,"+/-",np.std(diff)*c0/WM)



# compute barycentric correction
print('')
print('compute barycentric velocities')
gemini = EarthLocation.from_geodetic(lat=-30.2407*u.deg, lon=-70.7366*u.deg, height=2715*u.m) #IGRINS
sc = SkyCoord('20h45m09.5324974119s', '-31d20m27.237889841s') # AUMic
barycorr = sc.radial_velocity_correction('barycentric',obstime=Time(time_MJD,format='mjd'), location=gemini)
vbary = -barycorr.to(u.km/u.s).value


### Compute Planet-induced RV
Vp           = get_rvs(time_JD,Ks,P,t0)
Vc           = V0 + Vp - vbary   #Geocentric-to-barycentric correction



### Plot transit information
plot = True
if plot:
    print("\nPlot transit")
    TT     = 24.*(time_JD - t0)
    ypad   = 15  # pad of the y label
    plt.figure(figsize=(15,12))
    # Transit flux
    ax  = plt.subplot(411)
    ax.plot(TT,flux,"-+r")
    ax.set_ylabel("Transit curve\n", labelpad=ypad)
    # Airmass
    ax = plt.subplot(412)
    plt.plot(TT,airms,"-k")
    ax.set_ylabel("Airmass\n", labelpad=ypad)
    # RV correction between Geocentric frame and stellar rest frame
    ax = plt.subplot(413)
    plt.plot(TT,Vc,"-k")
    ax.set_ylabel("RV correction\n[km/s]", labelpad=ypad)
    # Maximum S/N
    ax = plt.subplot(414)
    plt.plot(TT,np.max(SN,axis=0),"+k")
    plt.axhline(np.mean(np.max(SN,axis=0)),ls="--",color="gray")
    plt.xlabel("Time wrt transit [h]")
    ax.set_ylabel("Peak S/N\n", labelpad=ypad)
    plt.subplots_adjust(hspace=0.02)
    plt.savefig("transit_info.pdf",bbox_inches="tight")
    plt.close()




orders   = np.arange(len(SN))
savedata = (orders,W_corr,I_corr,np.zeros_like(data_RAW),np.zeros_like(data_RAW),time_JD,phase,window,vbary,V0+Vp,airms,SN)
with open(name_fin, 'wb') as specfile:
    pickle.dump(savedata,specfile)
print('')
print('Finished.')
