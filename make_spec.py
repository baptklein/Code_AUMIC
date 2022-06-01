# -*- coding: utf-8 -*-
"""
Created on Nov 2020
@authors: Baptiste KLEIN
run with python3610 (or python368) on TITAN
"""

# ----------------------------------------------------------------------------------------------------------- #
# ---------------------------------- MAKE SPEC -------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------- #

### Goals:
#   Generate a 1D planet atmosphere template for each order using the petitRADTRANS python module
#   https://petitradtrans.readthedocs.io/en/latest/ 

### Inputs:
#   - Change the parameters below
#   - A file containing the list of [Order_nb Lamb_min Lamb_max Lamb_mean] used to compute velocity vector as
#     well as defining the limit for the computation of each order

### Modules
#   - Standard python modules: numpy and time
#   - petitRADTRANS: installed on titan (on a local folder)
#     To run the code and load the module on the titan computer cluster: 
#     1 - add "export PYTHONPATH=/data/atmo/petitRADTRANS-master/:$PYTHONPATH" to ~/.bashrc
#     2 - before running the code, always "source ~/.bashrc"
#     3 - Run code: "python368 make_spec.py"

#     Before using the code --> Create a repository called 'Model' to store the planet atmosphere template generated
#     WARNING: The planet atmosphere template are not normalized

# ----------------------------------------------------------------------------------------------------------- #



import numpy as np
import time
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc


t0 = time.time()
print("Init model")


# ----------------------------------------------------------------------------------------------------------- #
####### PARAMETERS -- USER INPUT
# ----------------------------------------------------------------------------------------------------------- #

Rp        = 1.186*nc.r_jup_mean     # Planet radius [cm]
Rs        = 5.28034* 10**(10)       # Stellar radius [cm]
gravity   = 2095.6                  # Planet surface gravity [cgs]
P0        = 0.01                    # Reference pressure [bar]
Teq       = 1200                    # Planet equilibrium temperature [K] -- Isothermal model
mmw       = 2.353                   # Mean molecular weight [cgs]
fact      = 0.01                   # Multiplicative factor to extend the wavelength ranges for petitRADTRANS
                                   # e.g., fact = 0.01 means that the wavelength range is 1% larger than the  min/max wavelengths in the input wave file


pressures                    = np.logspace(-10, 2, 130)           # Pressure grid [bar] 

### Abundances in MASSIVE FRACTIONS 
abundances                   =  {}
abundances['H2']             =  0.71 * np.ones_like(pressures)
abundances['He']             =  0.27 * np.ones_like(pressures)
abundances['H2O']            =  0.004 * np.ones_like(pressures)
abundances['CO_all_iso']     =  0.012* np.ones_like(pressures)
abundances['NH3']            =  2.2e-5* np.ones_like(pressures)
abundances['CO2']            =  1.0e-5* np.ones_like(pressures)
abundances['CH4']            =  3.4e-4* np.ones_like(pressures)
abundances['HCN']            =  1.1e-4* np.ones_like(pressures)

#abundances['CO_all_iso']     = 0.012* np.ones_like(pressures)
#abundances['NH3_main_iso']   = 2.2 * 10**(-5) * np.ones_like(pressures)
#abundances['CO2_main_iso']   = 1.0 * 10**(-5) * np.ones_like(pressures)
#abundances['CH4_main_iso']   = 3.4*10**(-4) * np.ones_like(pressures)
#abundances['HCN_main_iso']   = 1.1*10**(-4)* np.ones_like(pressures)


temperature  = Teq * np.ones_like(pressures)
MMW          = mmw * np.ones_like(temperature)

### Wavelengths
rep_wave   = "wave.dat"   ## Name of the input file listing the order nb, min/max/mean wavelengths for each order
A          = np.loadtxt(rep_wave,skiprows=1)
Num        = np.array(A[:,0],dtype=int)
W_min      = A[:,1]
W_max      = A[:,2]
Wm         = A[:,3]


### lbl species  --  for petitRADTRANS
#LS = ['H2O_main_iso','CO_all_iso','NH3_main_iso','CO2_main_iso','CH4_main_iso','HCN_main_iso']
LS        = ['CO_all_iso']




# ----------------------------------------------------------------------------------------------------------- #
####### MAIN PROCESS
# ----------------------------------------------------------------------------------------------------------- #

### Main loop - compute model and store results for each order

N_ord = len(Num)

for kk in range(N_ord):

    tx = "Order " + str(Num[kk])
    print(tx)
    
    lamb_inf,lamb_sup = (1-fact)*W_min[kk],W_max[kk]*(1+fact)

    ### make model
    atmosphere = Radtrans(line_species = LS, \
                      rayleigh_species = ['H2', 'He'], \
                      continuum_opacities = ['H2H2', 'H2He'], \
                      wlen_bords_micron = [lamb_inf/1000.,lamb_sup/1000.], \
                      mode = 'lbl')

    atmosphere.setup_opa_structure(pressures)

    atmosphere.calc_transm(temperature,abundances,gravity,MMW,\
                               R_pl=Rp,P0_bar=P0)
    
    ### Get the corresponding relative variation of flux (DF)
    Rp_l = atmosphere.transm_rad
    Wm   = nc.c/atmosphere.freq/1e-7 #nm
    DF   = -1.0* Rp_l**(2)/Rs**(2)

    ### Store the results
    rep_fin = 'Model/template_' + str(Num[kk]) + '.bin'

    file = open(rep_fin,'w')
    tx   = '### Model atmosphere HD189733b - Order ' + str(Num[kk]) + '\n'  
    tx  += '### Code: petitRADTRANS - 1D homogeneous isothermal atmosphere\n'
    tx  += '### Massive fractions: '

    for key in abundances:
        tx += str(key) + ": " + str(abundances[key][0]) + " "
    tx += "\n"

    tx  += '### T_eq = ' + str(Teq) +  ' K - mu = ' + str(mmw) + '\n'
    tx  += '### Wavelength  -Rp^2/Rs^2\n'
    file.write(tx)
    for nn in range(len(DF)):
        tx = str(Wm[nn]) + ' ' + str(DF[nn]) + '\n'
        file.write(tx)
    file.close()


t1  = time.time()
tx  = "DONE\n"
tx += "Total duration: " + str((t1-t0)/60.0) + " min"
print(tx)
