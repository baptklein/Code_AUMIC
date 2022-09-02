# -*- coding: utf-8 -*-
"""
Created in Sep 2021
Edited in Jun 2022
@authors: Baptiste KLEIN, Florian DEBRAS & Annabella MEECH
"""

import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import pickle
import scipy.interpolate as interp
import scipy.signal
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from functions import *
from scipy.stats import pearsonr
import astropy.constants as aC

def simple_correlation(list_ord,window,phase,Kp,Vtot,plot=False,savedir=None):
    """
    --> Inputs:     - list_ord:     List of Order objects (see "src.py")
                    - window:       Transit window
                    - phase:        Orbital phase for the planet
                    - Kp:           Grid of Kp values (1D vector)   --> semi-amplutde of planet RV signal
                    - Vtot:         Grid of total velocity values (1D vector)
                    - plot:         If True, will plot cross-correlation maps for each order
                    - savedir:      A directory in which to save plots (not saved if none set)

    --> Outputs:    - vsys_time:    Map of correlation coefficients between the observed and synthetic
                                spectra (2D matrix with time, velocity)

    """
    if plot and savedir is None:
        sys.exit('No directory defined for plots. Provide savedir.')
    ### INITIALISATION
    print("\nInitialization")
    data_tot = []
    wl       = []
    Stdtot   = []
    SNRtot   = []
    F        = []
    nord_tmp = len(list_ord)
    for kk in range(nord_tmp):
        wl.append(list_ord[kk].W_fin)                    # Store wavelength solution
        data_tot.append(list_ord[kk].I_pca)              # Store spectra (reduced, after PCA reduction)
        f = interp.interp1d(list_ord[kk].Wm,list_ord[kk].Im) # Interpolate the planet atmosphere template
        F.append(f)
    dv = Vtot[1]-Vtot[0]
    dph = np.average(phase[1:]-phase[:-1])

    # find epoch of start/end of transits
    n_ini     = 0
    while window[n_ini]==0.0: n_ini += 1
    n_end     = np.copy(n_ini)
    while window[n_end]!=0.0: n_end += 1
    print(n_ini,n_end)


    vsys_time = []
    vsys_kp   = []
    for no in range(len(list_ord)):
        ### insert data array here: time on first axis, wavelength on second: data[time, wavelength] ###
        data = data_tot[no]
        wavelengths = wl[no]

        # deal with nans
        data[np.isnan(data)] = 0.

        nep,npix = data.shape

        ### Do cross correlation ###
        vsys_time_ = np.zeros((nep, len(Vtot)))
        for i in range(len(Vtot)):
            for iep in range(nep):
                model_flux =  F[no](wavelengths[:] / (1 + (Vtot[i])/Constants().c0) )
                measured_flux = data[iep,:]
                vsys_time_[iep,i] = pearsonr(model_flux, measured_flux)[0]
        ### Stack up in time ###
        vsys_kp_ = np.zeros((len(Vtot), len(Kp)))
        ### Creating model of the cross correlation for each time ###
        f = [ interp.interp1d(Vtot, vsys_time_[k], kind='linear', fill_value='extrapolate') for k in range(nep) ]
        ### Obtaining cross correlation for kp vs vsys ###
        for j in range(len(Kp)):
        	for iep in range(nep):
        		vsys_kp_[:,j] += f[iep]( Vtot + Kp[j]*np.sin(2*np.pi*phase[iep]) )
        ### output ###
        vsys_time.append(vsys_time_)
        vsys_kp.append(vsys_kp_)
        ### plots ###
        if plot:
            nam_fig = savedir + 'corr_vel_map_order{}.png'.format(no+1)
            plt.figure(figsize=(12,4))
            extent = (Vtot.min()-0.5*dv,Vtot.max()-0.5*dv,phase.max()-0.5*dph,phase.min()-0.5*dph)
            plt.imshow(vsys_time_,extent=extent,aspect='auto')
            plt.colorbar()
            plt.xlabel('Velocity (km/s)')
            plt.ylabel('Orbital phase')
            plt.axhline(phase[n_ini],ls=':',color="k")
            plt.axhline(phase[n_end],ls=':',color="k")
            plt.savefig(nam_fig,bbox_inches="tight")
            plt.close()
    return vsys_time,vsys_kp
# -----------------------------------------------------------
# Compute the correlation map in the (Kp,Vsys) space
# For each SPIRou order, compute the correlation coefficient
# between the reduced sequence of spectra and a sequence of
# synthetic spectra shifted according to the planet RV on a (Kp,Vsys)
# grid.
# To gain time in the correlation process, we pre-compute a grid
# of sequence of synthetic spectra that we use in the correlation
# computation.
# -----------------------------------------------------------
def compute_correlation(list_ord,window,phase,Kp,Vsys,V_shift):

    """
    --> Inputs:     - list_ord: List of Order objects (see "src.py")
                    - window:   Transit window
                    - phase:    Orbital phase for the planet
                    - Kp:       Grid of Kp values (1D vector)   --> semi-amplutde of planet RV signal
                    - Vsys:     Grid of Vsys values (1D vector) --> RV of the planet at mid-transit

    --> Outputs:    - correl:   Map of correlation coefficients between the observed and synthetic
                                spectra (2D matrix with N_Kp, N_Vsys - one for each order)

    """

    t0           = time.time()
    c0           = Constants().c0
    pixel_window = np.linspace(-1.14,1.14,10)          ### grid to integrate interpolated model over a SPIRou pixel
    weights      = scipy.signal.gaussian(10,std=1.14)  ### Gaussian profile centered on 0.0 (for pixel integration)

    ### INITIALISATION
    print("\nInitialization")
    data_tot = []
    wl       = []
    Stdtot   = []
    SNRtot   = []
    F        = []
    nord_tmp = len(list_ord)
    for kk in range(nord_tmp):                           # For each order:
        std_px = list_ord[kk].I_pca.std(axis=0)          # Dispersion of each pixel along the time axis
        fit_px = np.polyfit(list_ord[kk].W_fin,std_px,2) # Fit polynomial of order 2 to the distribution (blaze function)
        Stdfit = np.poly1d(fit_px)
        Stdtot.append(Stdfit(list_ord[kk].W_fin))        # Store best-fitting solution
        SNRtot.append(list_ord[kk].SNR)
        wl.append(list_ord[kk].W_fin)                    # Store wavelength solution
        data   = list_ord[kk].I_pca
        data[np.isnan(data)] = 0.                        # rid nans
        data_tot.append(data)                            # Store spectra (reduced, after PCA reduction)
        f = interp.interp1d(list_ord[kk].Wm,list_ord[kk].Im) # Interpolate the planet atmosphere template
        F.append(f)




    # In order to gain time in the correlation process, we pre-compute the grid of models at the resolution of the data
    print("Interpolate model")

    gri      = []
    for i in range(len(Kp)):
        for j in range(len(Vsys)):
            gri.append(Kp[i]*np.sin(2.0*np.pi*np.array(phase,dtype=float))+Vsys[j] + np.array(V_shift,dtype=float))
    gri      = np.array(gri,dtype=float)
    MM       = np.max([np.abs(gri.min()),np.abs(gri.max())])
    Vtot     = np.linspace(-1.01*MM,1.01*MM,500)


    F2D      = []
    for i in range(nord_tmp):
        mod_int = np.zeros((len(Vtot),len(wl[i])))
        for j in range(len(Vtot)):
            mod_int[j]= np.average(F[i](list(map(lambda x: wl[i]/(1.0+(Vtot[j]+x)/c0),pixel_window))),weights=weights,axis=0)
        f2D = interp.interp1d(Vtot,mod_int.T,kind='linear')
        F2D.append(f2D)


    ### Increase the speed of the correlation
    pos     = np.where(np.array(window)>0.5)[0]
    phase2  = np.array(phase)[pos]
    window2 = np.array(window)[pos]
    Vshift2 = np.array(V_shift)[pos]
    correl  = np.zeros((len(Kp),len(Vsys),len(list_ord),len(phase2)))


    #And now the correlation, following boucher et al. 2021
    print("Compute correlation for",nord_tmp,"orders")
    nbor = 50 # remove edge pixels
    for ii in range(len(Kp)):
        for jj in range(len(Vsys)):
            vp             = Kp[ii]*np.sin(2.0*np.pi*phase2)+Vsys[jj] + Vshift2
            for no in range(len(list_ord)):
                interpmod      = F2D[no](vp)
                modelij        = interpmod[nbor:-nbor]-np.mean(interpmod[nbor:-nbor],axis=0)
                dataij         = (data_tot[no][pos][:,nbor:-nbor].T-np.mean(data_tot[no][pos][:,nbor:-nbor].T,axis=0))
                correl[ii,jj,no] = np.sum(((dataij*modelij*window2/SNRtot[no][pos]**2).T*np.mean(SNRtot[no][pos]**2)/Stdtot[no][nbor:-nbor]**2).T,axis=0)


    print("DONE!")
    t1 = time.time()
    print("Duration:",(t1-t0)/60.,"min")

    return correl



# -----------------------------------------------------------
# Compute the typical noise dispersion in a correlation map:
# (i)   Combine the correlation maps associated to an input list of orders
# (ii)  Select a region where "no planetary signal is present" from the
#       combined correlation map
# (iii) Compute and return the dispersion of the selected map
# -----------------------------------------------------------
def get_snrmap(orders_fin,Kp,Vsys,correl,Kp_lim=[120,180],Vsys_lim=[-15.,15.]):

    """
    --> Inputs:     - orders_fin: list of orders to combine
                    - Kp: Grid of Kp (1D vector)
                    - Vsys: Grid of Vsys (1D vector)
                    - correl: Correlation map
                    - Kp_lim: Kp range to be excluded when compute the noise level (list of 2 values)
                    - Vsys_lim: Vsys range to be excluded when compute the noise level (list of 2 values)

    --> Outputs:    - snrmap: Estimated noise dispersion in the correlation map

    """

    # Select orders to combine
    sel = []
    for i in orders_fin:
        sel.append(np.where(np.array(orders_fin,dtype=int)==i)[0][0])
    a      = np.sum(np.sum(correl[:,:,sel],axis=3),axis=2)                     # Get correlation map for the orders
    b      = a[np.where((Kp<Kp_lim[0])|(Kp>Kp_lim[1]))]            # Exclude Kp range
    c      = b.T[np.where((Vsys<Vsys_lim[0])|(Vsys>Vsys_lim[1]))]  # Exclude Vsys range
    snrmap = np.std(c)                                             # Compute disperion
    return snrmap





# -----------------------------------------------------------
# Compute a bivariate normal law on (X,Y) input vectors
# -----------------------------------------------------------
def multi_var(param,X,Y):

    """
    --> Inputs:     - param: vector of parameters for the bivariate normal law
                      param[0] --> amplitude
                      param[1] --> Mean of X
                      param[2] --> Mean of Y
                      param[3] --> STD of X
                      param[4] --> STD of Y
                      param[5] --> COV(X,Y)
                    - X: Input vector X
                    - Y: Input vector Y


    --> Outputs:    - 2D normal law in the (X,Y) space

    """

    amp = param[0]
    mu  = np.array([param[1],param[2]],dtype=float)
    cov = np.array([[param[3]**(2),param[5]],[param[5],param[4]**(2)]],dtype=float)
    pv  = multivariate_normal(mu,cov)
    pos = np.dstack((X,Y))
    mv  = amp*pv.pdf(pos)
    return np.array(mv,dtype=float)






# -----------------------------------------------------------
# Function to minimize when fitting a bivariate normal law
# -----------------------------------------------------------
def crit(param,X,Y,C):

    """
    --> Inputs:     - param: vector of parameters for the bivariate normal law
                      param[0] --> amplitude
                      param[1] --> Mean of X
                      param[2] --> Mean of Y
                      param[3] --> STD of X
                      param[4] --> STD of Y
                      param[5] --> COV(X,Y)
                    - X: Input vector X
                    - Y: Input vector Y
                    - C: Data

    --> Outputs:    - Norm 2 of the residuals between observed map C and 2D bivariate Gaussian map for param, X and Y

    """
    mv = multi_var(param,X,Y)
    cr = np.linalg.norm(C-mv) ### 2-norm
    return cr



# -----------------------------------------------------------
# Fit a 2D bivariate normal distribution to an observed map C
# -----------------------------------------------------------
def fit_multivar(x,y,C):

    """
    --> Inputs:     - x: X vector (e.g. Vsys)
                    - y: Y vector (e.g. Kp)
                    - C: 2D map to fit (e.g. correlation map)

    --> Outputs:    - p_best: best set of parameters (see the parameter description in "crit" function
                    - mv_best: Best-fitting biariate normal law (2D matrix in the (x,y) plane)

    """

    # Inital set of parameters
    sigma_x   = 2.0
    sigma_y   = 20.0
    sigma_xy  = 0.0
    mu_x      = 0.0
    mu_y      = 150.0
    amp       = 750.0
    param0    = [amp,mu_x,mu_y,sigma_x,sigma_y,sigma_xy]
    X_s,Y_s   = np.meshgrid(x,y)
    print("Fit bivariate normal law on significance map")
    res       = minimize(crit,param0,args=(X_s,Y_s,C),method="Nelder-Mead") # Fit bivariate normal distribution
    p_best    = res.x
    mv_best   = multi_var(p_best,X_s,Y_s)
    return p_best,mv_best





# -----------------------------------------------------------
# Get statistics on an input significance map:
# - Compute best estimates of Kp and Vsys
# - Estimate maximum significance
# - First rough estimate of errorbars on Kp and Vsys
# -----------------------------------------------------------
def get_statistics(x,y,C):

    """
    --> Inputs:     - x: X vector (e.g. Vsys)
                    - y: Y vector (e.g. Kp)
                    - C: 2D map to fit (e.g. correlation map)

    --> Outputs:    - p_best: best set of parameters (see the parameter description in "crit" function
                    - Kp_best: Best estimate of Kp [km/s)
                    - K_sup: Upper uncertainty on Kp [km/s]
                    - K_inf: Lower uncertainty on Kp [km/s]
                    - V_best: Best estimate on Vsys
                    - V_sup: Upper uncertainty on Vsys [km/s]
                    - V_inf: Lower uncertainty on Vsys [km/s]

    """

    # Start by fitting a bivariate normal law on the significance map C
    p_best,mv_best = fit_multivar(x,y,C)

    # Get best-fitting parameters
    V_best  = p_best[1]
    Kp_best = p_best[2]

    # Create densely-sampled grids of Kp an Vsys
    # and make densely-sampled 2D bivariate normal law
    V_big   =  np.linspace(x.min(),x.max(),101)
    Kp_big  =  np.linspace(y.min(),y.max(),101)
    Xb,Yb   = np.meshgrid(V_big,Kp_big)
    mv_tot  = multi_var(p_best,Xb,Yb)  # Finely-sampled bivariate law

    # Max significance
    sn_max = mv_tot.max()
    print("Maximum detection:",round(sn_max,1),"sigma")

    # Compute error bars on Kp and Vsys:
    # Start from the best-fitting value and take upper/lower value along Kp (at best Vsys) and Vsys (at best Kp)
    # such that the significance decreases by 1 sigma
    # NOTE: If the grid we cannot reach a difference of 1 sigma between best parameter (grid too small), we print
    # a warning and return the size of the grid as error bars
    ib1         = np.argmin(np.abs(V_big-V_best))
    ib2         = np.argmin(np.abs(Kp_big-Kp_best))
    K_sup,K_inf = 0.0,0.0
    V_sup,V_inf = 0.0,0.0
    ii    = ib2
    while mv_tot[ii,ib1]>sn_max-1.0 and ii<len(Kp_big)-1: ii+=1
    if ii == len(Kp_big)-1: print("\n\nWARNING -- Window too small -- Underestimation of error bars\n\n")
    K_sup = Kp_big[ii]-Kp_best
    ii    = ib2
    while mv_tot[ii,ib1]>sn_max-1.0 and ii>0: ii-=1
    if ii == 0: print("\n\nWARNING -- Window too small -- Underestimation of error bars\n\n")
    K_inf = Kp_best-Kp_big[ii]
    ii    = ib1
    while mv_tot[ib2,ii]>sn_max-1.0 and ii<len(V_big)-1: ii+=1
    if ii == len(V_big)-1: print("\n\nWARNING -- Window too small -- Underestimation of error bars\n\n")
    V_sup = V_big[ii]-V_best
    ii    = ib1
    while mv_tot[ib2,ii] > sn_max-1.0 and ii>0: ii -= 1
    if ii == 0: print("\n\nWARNING -- Window too small -- Underestimation of error bars\n\n")
    V_inf = V_best-V_big[ii]

    # Display results
    print("Best estimates:")
    print("Kp:",round(Kp_best,1),"(+",round(K_sup,1),",-",round(K_inf,1),") km/s")
    print("V0:",round(V_best,1),"(+",round(V_sup,1),",-",round(V_inf,1),") km/s")

    return p_best,Kp_best,K_sup,K_inf,V_best,V_sup,V_inf



# -----------------------------------------------------------
# Plot correlation/significance map in the (Kp,Vsys) space.
# We give 2 options:
# - Either simple color map (if sn_cutx = [])
# - Either color map + 1D cuts at best Kp and Vsys
# -----------------------------------------------------------
def plot_correlation_map(Vsys,Kp,sn_map,nam_fig,V_inj=0.0,K_inj=0.0,cmap="gist_heat",sn_cutx=[],sn_cuty=[],levels=10):

    ### Simple color maps
    if len(sn_cutx) == 0:
        plt.figure(figsize=(10,7))
        plt.contourf(Vsys,Kp,sn_map,levels=levels,cmap=cmap)
        plt.ylabel(r"K$_{\rm{p}}$ [km/s]")
        plt.xlabel(r"V$_{\rm{sys}}$ [km/s]")
        plt.axhline(K_inj,ls="--",lw=1.0,color="w")
        plt.axvline(V_inj,ls="--",lw=1.0,color="w")
        cb = plt.colorbar()
        cb.set_label(r"Significance [$\sigma$]",rotation=270,labelpad=40)
        plt.savefig(nam_fig,bbox_inches="tight")
        plt.close()

    ### Color maps with 1D cuts at best Kp and Vsys
    else:
        # Create grid of subplots
        fig     = plt.figure(figsize=(12,12))
        grid    = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1)
        main_ax = fig.add_subplot(grid[1:,:3])
        y_hist  = fig.add_subplot(grid[1:,-1], yticklabels=[])
        x_hist  = fig.add_subplot(grid[:1, :3], xticklabels=[])

        # Color map
        main_ax.contourf(Vsys,Kp,sn_map,levels=levels,cmap=cmap)
        main_ax.axhline(K_inj,ls=":",color="w",lw=2.5)
        main_ax.axvline(V_inj,ls=":",color="w",lw=2.5)
        main_ax.set_ylabel(r"K$_{\rm{p}}$ [km/s]")
        main_ax.set_xlabel(r"V$_{\rm{sys}}$ [km/s]")

        # Cut at best Kp
        x_hist.plot(Vsys,sn_cuty,"-k")
        x_hist.axhline(0.0,ls=":",color="gray")
        x_hist.axvline(V_inj,ls=":",color="r")
        title_x = r"Cut at K$_{\rm{p}}$ = " + str(K_inj) + " km/s"
        x_hist.set_xlabel(title_x,rotation=0,labelpad=-180)
        x_hist.set_ylabel(r"Significance [$\sigma$]")

        # Cut at best Vsys
        y_hist.plot(sn_cutx,Kp,"-k")
        y_hist.axvline(0.0,ls=":",color="gray")
        y_hist.axhline(K_inj,ls=":",color="r")
        title_y = r"Cut at V$_{\rm{sys}}$ = " + str(V_inj) + " km/s"
        y_hist.set_ylabel(title_y,rotation=270,labelpad=-170)
        y_hist.set_xlabel(r"Significance [$\sigma$]")

        plt.savefig(nam_fig,bbox_inches="tight")
        plt.close()

# -----------------------------------------------------------
# Plot correlation/time map for a single order.
# - filename: path to cross-correlation pkl result file (from get_correl.py)
# - order: index of the order to plot
# - vsys_cut: the systemic velocity to cut through the velocity map (km/s)
# -----------------------------------------------------------
def cc_time_plot(filename,order,vsys_cut):
    # Read in the cross-correlation result
    print("Read data from",filename+'.pkl')
    res_dir = 'xcorr_result/' + filename + '.pkl'
    with open(res_dir,'rb') as ccfile:
        Vsys,Kp,corr,sn_map = pickle.load(ccfile)
    # corr is the result from compute_correlation(), ie just in-transit
    nKp,nVsys,nord,nep = corr.shape
    Kpmin, Kpmax    = Kp.min(),Kp.max()
    dKp = Kp[1]-Kp[0]

    if len(vsys_cut)>1:
        # sum over a range
        v1,v2 = vsys_cut
        res = np.sum(corr[:,v1:v2,order,:],axis=1)
    else:
        vsys_cut = vsys_cut[0]
        res = corr[:,vsys_cut,order,:]

    fig     = plt.figure(figsize=(15,0.05*nep))
    extent  = (Kpmin - 0.5*dKp, Kpmax -0.5*dKp, nep-0.5, 0.5)
    plt.imshow(res,aspect='auto',interpolation='nearest',extent=extent)
    plt.xlabel('Kp (km/s)')
    plt.ylabel('Frame number (in transit!)')
    plt.title('Model: ' +filename+', Order: {}'.format(order)+', Vsys_cut: {}'.format(vsys_cut))
    plt.colorbar()
    plt.savefig('xcorr_result/' + filename + '_order{}.png'.format(order))
    plt.close()
