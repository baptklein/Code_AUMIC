#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Mar 2022
Edited in Jun 2022

@author: Baptiste KLEIN & Annabella MEECH
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

dir_data = "Data/IGRINS/"   #Sep18 #Jun19

# define planet b parameters
# from Martioli+2021
P        = 8.463000 # period in days
t0       = 2458330.39051 # known mid-transit time in JD
Rp       = 25929.97  #Planet radius  [km]
Rs       = 522255.0  #Stellar radius [km]
ip       = 89.5      #Transit incl.  [deg]
ap       = 18.476     #Semi-maj axis  [R_star]
ep       = 0.0     #Eccentricity of Pl. orbit
wp       = 90.0     #Arg of periaps [deg]
### Get the limb-darkening coefficients in H band from Claret+2011: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/529/A75
ld_mod   = "nonlinear"     #Limb-darkening model ["nonlinear", "quadratic", "linear"]
ld_coef  = [1.2783,-1.5039,1.2042,-0.3833] # Claret et al. 2011

### Stellar radial velocity info
Ks        = 0.006    #RV semi-amplitude of the star orbital motion due to planet [km/s]
V0        = -4.71    #Stellar systemic velocity [km/s]



c0 = Constants().c0

### Name of the pickle file to store the info in
outroot = "Input_data/igrins/"
os.makedirs(outroot,exist_ok=True)
name_fin = outroot+"data_igrins.pkl"
wlen_fin = outroot+'wlens_igrins.pkl'
humidity_fin = outroot+'humidity_igrins.pkl'


time_JD, wlens, data_RAW, data_var, data_sn, airms, humidity = read_igrins_data(dir_data)


plt.plot(time_JD,"+")
plt.title('JD')
plt.show()



tmid = t0
i = 0
while tmid<time_JD[0]:
    tmid += P
    i += 1
print('')
print('mid-transit time: {}'.format(tmid))


### Compute transit window
### Using batman python package https://lweb.cfa.harvard.edu/~lkreidberg/batman/
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



for nn in range(len(W_corr)):
    WW   = W_corr[nn]
    WM   = np.mean(WW)
    diff = np.diff(WW)
    print(np.mean(diff)*c0/WM,"+/-",np.std(diff)*c0/WM)



# compute barycentric correction
print("\ncompute barycentric velocities")
gemini = EarthLocation.from_geodetic(lat=-30.2407*u.deg, lon=-70.7366*u.deg, height=2715*u.m) #IGRINS
sc = SkyCoord('20h45m09.5324974119s', '-31d20m27.237889841s') # AUMic
barycorr = sc.radial_velocity_correction('barycentric',obstime=Time(time_JD,format='jd'), location=gemini)
vbary = -barycorr.to(u.km/u.s).value


### Compute Planet-induced RV
Vp           = get_rvs(time_JD,Ks,P,t0)
Vc           = V0 + Vp - vbary   #Geocentric-to-barycentric correction
# V0 is the stellar systemic velocity
# Vp is the RV of the star due to the planet


### Plot transit information
plot = True
if plot:
    print("\nplot transit")
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
# save wlens in a separate file
with open(wlen_fin, 'wb') as wlenfile:
    pickle.dump(W_corr,wlenfile)
# save humidity in a separate file
with open(humidity_fin, 'wb') as humfile:
    pickle.dump(humidity,humfile)
print("\nFINISHED.")
