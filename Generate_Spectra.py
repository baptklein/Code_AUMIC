"""
Created in Aug 2022
@author: Oliver PEARCE, Baptiste KLEIN & Annabella MEECH
"""

# This code takes existing data (generate using calculate_atmosphere.py) and makes various plots
# If the data doesn't already exist then this calls the calculate_atmosphere.py file to generate the data first, then plot
# (Exception: the data must exist for the contribution_spectrum plot

# Important: Make sure to comment out the running of any functions in Calculate_Atmosphere.py to prevent running the script here.
from Calculate_Atmosphere import *

import matplotlib
font = {'size'   : 18,
        'weight': 'light'}
axes = {'labelsize'   : 18}
matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)

# INPUT: also see further inputs later in code e.g. output directories

# Input atmospheric metallicities of planet to produce spectra for, in units of solar metallicity:
metallicities = [1,10,100]

# Input atmospheric C/O ratios to produce spectra for:
C_O_ratios = [0.5,0.6,0.7,0.8,0.9,1.0]


### ---- generate_spectrum function ---- ###

# Plot transmission spectrum of any combination of species. If data does not exist for that combination/wavelength range, it shall ran the calculate_atmosphere function in
# Calculate_Atmosphere.py to calculate the data, then produce the spectra.
#
# INPUTS:
# min/max_wavelength: wavelength bounds (nm) over which to test.
# orders = 'yes' or 'no': choose whether to overlay the spirou diffraction grating orders on the spectrum - default is yes.
# species = []: A list containing the species to produce the spectra for - default is all species: ['CH4', 'CO', 'CO2', 'H2O', 'NH3']
def generate_spectrum(min_wavelength, max_wavelength, orders='yes', species=['CH4', 'CO', 'CO2', 'H2O', 'NH3'], haze_factor='no', Pcloud=None):
    for metallicity in metallicities:
        for ratio in C_O_ratios:
            # Check if output directory exists, create it if not
            # INPUT: Your path to output directory
            output_dir = 'Models/' + str(metallicity) + 'x_metallicity_' + str(ratio) + '_CO_ratio/' +\
                         str(int(min_wavelength)) + '_' + str(int(max_wavelength)) + '/'
            os.makedirs(output_dir, exist_ok=True)

            if len(species)==1:
                name = species[0]
            else:
                species.sort()
                name = '_'.join(species)

            wavelength = []
            normalised_flux = []

            # Checks if file exists, loads data if so
            path_to_file = output_dir + 'pRT_data_' + name + '.dat'
            path = Path(path_to_file)
            if path.is_file():
                with open(path_to_file, 'r') as data:
                    lines = data.readlines()
                    data.close()

            # If file doesn't exist, calculates the atmosphere for this combination
            else:
                print("No data currently exists for this combination of species/wavelength range, calculating this atmosphere now")
                pRT_name = []
                for i in range(len(species)):
                    pRT_index = plot_species_lables.index(species[i])
                    pRT_name.append(pRT_names[pRT_index])


                calculate_atm(pRT_name, min_wavelength=min_wavelength, max_wavelength=max_wavelength,metallicities=[metallicity],
                              C_O_ratios=[ratio], haze_factor=haze_factor, Pcloud=Pcloud)
                with open(path_to_file, 'r') as data:
                    lines = data.readlines()
                    data.close()

            # Read data from file, and gathers values to plot
            for line in lines[4:]:
                v = line.split(' ')
                wavelength.append(float(v[0]))
                normalised_flux.append(float(v[1].split('\n')[0]))

            plt.figure(figsize=[20, 12])
            plt.plot(wavelength, normalised_flux, label='Spectrum')

            # Plotting the diffraction grating orders if selected
            if orders == 'yes' or orders == 'Yes' or orders == 'Y':
                spirou_min = []
                spirou_max = []
                spirou = open("orders_spirou.dat", "r")
                spirou.readline()
                for line in spirou:
                    values = line.split()
                    spirou_min.append(values[2])
                    spirou_max.append(values[3])

            # Ensuring that for each plot only the orders within the specified wavelength range are gathered.
            # NOTE -- The orders do overlap, so the max and min adjacent to each other don't necessarily define an order,
            # the orders typically cover ~50nm.

                wavelength_min = np.amin(wavelength)
                wavelength_max = np.amax(wavelength)
                orders_min = []
                orders_max = []

                for value in spirou_min:
                    if float(value) > wavelength_min and float(value) < wavelength_max:
                        orders_min.append(float(value))
                for value in spirou_max:
                    if float(value) > wavelength_min and float(value) < wavelength_max:
                        orders_max.append(float(value))

                plt.vlines(orders_min, ymin=np.amin(normalised_flux), ymax=np.amax(normalised_flux), colors='red',
                           linestyles='dashed', label='Minimum of order')
                plt.vlines(orders_max, ymin=np.amin(normalised_flux), ymax=np.amax(normalised_flux), colors='green',
                           linestyles='dashed', label='Maximum of order')

                plt.title(
                    'Normalised Transmission Spectrum (showing diffraction grating orders) of ' + name + ' at ' + str(
                        metallicity) + 'x solar metallicity, ' + str(ratio)
                    + ' C/O ratio')

            elif orders == 'no' or orders == 'No':
                plt.title('Normalised Transmission Spectrum of ' + name + ' at ' + str(
                    metallicity) + 'x solar metallicity, ' + str(ratio)
                          + ' C/O ratio')

            plt.xlabel('Wavelength (nm)')
            plt.ylabel(r'Normalised Flux')
            plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
            plt.legend()
            # Saving spectrum
            file = Path(output_dir + 'Spectrum_' + name + '.pdf')
            file.touch(exist_ok=True)
            plt.savefig(file)
            plt.clf()


### ---- contribution_spectrum function ---- ###

# Plots an overall transmission spectrum (for all species)and also the contribution of the specified individual species that make up the full spectrum.
# By default the full spectrum is for CH4, CO, CO2, H2O and NH3 - as this is the full atmosphere I have calculated. You can change this below but then you need to
# run the calculate_atmosphere function for that combination to generate the data.
#
# INPUTS:
# min/max_wavelength: wavelength bounds (nm) over which to test.
# species = []: A list containing the species to plot the individual spectrum of (on top of the overall) - default is all species: ['CH4', 'CO', 'CO2', 'H2O', 'NH3']
def contribution_spectrum(min_wavelength, max_wavelength, species=['CH4', 'CO', 'CO2', 'H2O', 'NH3']):
    for metallicity in metallicities:
        for ratio in C_O_ratios:

            # INPUT: Your path to output directory
            output_dir = 'Models/' + str(metallicity) + 'x_metallicity_' + str(ratio) + '_CO_ratio/' + \
                         str(int(min_wavelength)) + '_' + str(int(max_wavelength)) + '/'
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=[15, 9])

            # Plotting full spectrum - CH4, CO, CO2, H2O AND NH3 by default: - CHANGE IF DESIRED THOUGH ENSURE DATA EXISTS FOR COMBINATION
            with open(output_dir+'pRT_data_CH4_CO_CO2_H2O_NH3.dat', 'r') as data:
                wavelength = []
                normalised_flux = []
                lines = data.readlines()
                data.close()
                name = lines[0].split(': ')[1]
                molecules = name.split('\n')[0]

            # Plotting overall spectrum
            for line in lines[4:]:
                v = line.split(' ')
                wavelength.append(float(v[0]))
                normalised_flux.append(float(v[1]))
            av_flux=np.average(normalised_flux)
            plt.plot(wavelength, normalised_flux, label='Overall Spectrum', zorder=av_flux, alpha=1)

            # Plotting spectrum of individual species specified in the species input
            for molecule in species:
                with open(output_dir+'pRT_data_'+molecule+'.dat') as data:
                    wavelength = []
                    normalised_flux = []
                    lines = data.readlines()
                    data.close()
                for line in lines[4:]:
                    v = line.split(' ')
                    wavelength.append(float(v[0]))
                    normalised_flux.append(float(v[1]))
                # using the flux to define plotting order
                av_flux = np.average(normalised_flux)
                plt.plot(wavelength, normalised_flux, label=molecule, zorder=av_flux, alpha=1)


            plt.xlabel('Wavelength (nm)')
            plt.ylabel(r'Normalised Flux')
            plt.legend()
            plt.title('Normalised Transmission Spectrum showing contribution of individual species at '+str(metallicity)+'x solar metallicity, '+str(ratio)
                              +' C/O ratio', wrap=True)
            plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
            file = Path(output_dir + 'Contribution_Spectrum_' + molecules + '.pdf')
            file.touch(exist_ok=True)
            plt.savefig(file)
            plt.clf()


### ---- varying_metallicity function ---- ###
#
# Plot the transmission spectra of a species/combination of species at a given C/O ratio but varying metallicities. If data does not exist for that combination/wavelength range, it shall ran the calculate_atmosphere function in
# Calculate_Atmosphere.py to calculate the data, then produce the spectra.
#
# INPUTS:
# ratio: the C/O ratio at which to plot the varying metallicity data for.
# min/max_wavelength: wavelength bounds (nm) over which to test.
# species = []: A list containing the species to produce the spectra for - default is all species: ['CH4', 'CO', 'CO2', 'H2O', 'NH3']
def varying_metallicity(ratio, min_wavelength, max_wavelength, species=['CH4', 'CO', 'CO2', 'H2O', 'NH3']):

    # # INPUT: Your path to input and output directory
    input_dir = 'Models/'
    output_dir = 'Models/Varying_Metallicity/'+str(int(min_wavelength)) + '_' + str(int(max_wavelength)) + '/'
    # check if output directory exists, create it if it doesn't
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=[15,9])
    if len(species) == 1:
        name = species[0]
    else:
        species.sort()
        name = '_'.join(species)

    # Checking if data exists...
    sub_dirs = [f for f in Path(input_dir).iterdir() if f.is_dir()]
    for i in sub_dirs:
        if str(ratio) in str(i):
            folder = str(i) + '/' + str(min_wavelength) + '_' + str(max_wavelength) + '/'
            path = Path(folder+'pRT_data_' + name+'.dat')
            if path.is_file():
                with open(path, 'r') as data:
                    lines = data.readlines()
                    data.close()
            # ...calculating if not
            else:
                print("No data currently exists for this combination of species/wavelength range, calculating this atmosphere now")
                pRT_name = []
                for i in range(len(species)):
                    pRT_index = plot_species_lables.index(species[i])
                    pRT_name.append(pRT_names[pRT_index])

                # Default testing all metallicities, change if necessary
                calculate_atm(pRT_name, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                              metallicities=[1, 10, 100, 1000, 10000],
                              C_O_ratios=[ratio])
                plt.figure(figsize=[15,9])
                with open(path, 'r') as data:
                    lines = data.readlines()
                    data.close()

            # Plotting - zorder defines plotting order, taking the spectrum with the strongest spectrum to appear on bottom and others on top so that it is easier to interpret
            wavelength = []
            normalised_flux = []
            metallicity = lines[1].split(' ')[1]
            for line in lines[4:]:
                v = line.split(' ')
                wavelength.append(float(v[0]))
                normalised_flux.append(float(v[1]))
            min_flux = np.amin(normalised_flux)
            plt.plot(wavelength, normalised_flux, label=str(metallicity), zorder=min_flux, alpha=1)

        else:
            continue

    plt.xlabel('Wavelength (nm)')
    plt.ylabel(r'Normalised Flux')
    current_handles, current_labels = plt.gca().get_legend_handles_labels()
    ordered_handles = [element for _, element in sorted(zip(current_labels, current_handles))]
    current_labels.sort()
    plt.legend(ordered_handles, current_labels, prop={'size': 11}, title='Metallicity \n(x solar value)', title_fontsize=11)
    plt.title('Normalised Transmission spectra of ' + name + ' at ' + str(ratio) + ' C/O ratio and varying metallicities', wrap=True)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    file = Path(output_dir +'C_O_'+str(ratio) + '_' + name + '.pdf')
    file.touch(exist_ok=True)
    plt.savefig(file)
    plt.clf()


### ---- varying_ratio function ---- ###
#
# Plot the transmission spectra of a species/combination of species at a given metallicity but varying C/O ratios. If data does not exist for that combination/wavelength range, it shall ran the calculate_atmosphere function in
# Calculate_Atmosphere.py to calculate the data, then produce the spectra.
#
# INPUTS:
# metallicity: the metallicity at which to produce the spectra for
# min/max_wavelength: wavelength bounds (nm) over which to test.
# species = []: A list containing the species to produce the spectra for - default is all species: ['CH4', 'CO', 'CO2', 'H2O', 'NH3']

def varying_ratio(metallicity, min_wavelength, max_wavelength, species=['CH4', 'CO', 'CO2', 'H2O', 'NH3']):

    # INPUT: Your path to input and output directory
    input_dir = 'Models/'
    output_dir = 'Models/Varying_Ratio/'+str(int(min_wavelength)) + '_' + str(int(max_wavelength)) + '/'
    # check if output directory exists, create it if it doesn't
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=[15, 9])
    if len(species) == 1:
        name = species[0]
    else:
        species.sort()
        name = '_'.join(species)

    # Checking if fles exist...
    sub_dirs = [f for f in Path(input_dir).iterdir() if f.is_dir()]
    for i in sub_dirs:
        if str(metallicity)+'x' in str(i):
            folder = str(i) + '/' + str(min_wavelength) + '_' + str(max_wavelength) + '/'
            path = Path(folder + 'pRT_data_' + name + '.dat')
            if path.is_file():
                with open(path, 'r') as data:
                    lines = data.readlines()
                    data.close()
            # ...calculating the data if not
            else:
                print("No data currently exists for this combination of species/wavelength range, calculating this atmosphere now")
                pRT_name = []
                for i in range(len(species)):
                    pRT_index = plot_species_lables.index(species[i])
                    pRT_name.append(pRT_names[pRT_index])

                # Default testing all metallicities, change if necessary
                calculate_atm(pRT_name, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                              metallicities=[metallicity],
                              C_O_ratios=[0.5, 1.0, 1.5])
                with open(path, 'r') as data:
                    lines = data.readlines()
                    data.close()

            wavelength = []
            normalised_flux = []
            ratio = lines[2].split(' ')[1]
            for line in lines[4:]:
                v = line.split(' ')
                wavelength.append(float(v[0]))
                normalised_flux.append(float(v[1]))
            min_flux = np.amin(normalised_flux)

            plt.plot(wavelength, normalised_flux, label=str(ratio), zorder=min_flux, alpha=1)

        else:
            continue

    plt.xlabel('Wavelength (nm)')
    plt.ylabel(r'Normalised Flux')
    current_handles, current_labels = plt.gca().get_legend_handles_labels()
    ordered_handles = [element for _, element in sorted(zip(current_labels, current_handles))]
    current_labels.sort()
    plt.legend(ordered_handles, current_labels, prop={'size': 11}, title='C/O ratio',
               title_fontsize=11)
    plt.title(
        'Normalised Transmission spectra of ' + name + ' at ' + str(metallicity) + 'x solar metallicity and varying C/O ratio',
        wrap=True)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    file = Path(output_dir + str(metallicity) + 'x_Metallicity_' + name + '.pdf')
    file.touch(exist_ok=True)
    plt.savefig(file)
    plt.clf()


### --- clear_cloudy function -- ####
# Plots spectra for the same conditions and species, but with both a clear and a hazy model
# Input min and max wavelength, and the species you want to plot for
# If no cloudy data exists (assumes the clear data exists) it will calculate the hazy atmosphere for default haze factor of 10

def clear_cloudy(min_wavelength, max_wavelength, species=['CO', 'CH4', 'H2O', 'NH3', 'CO2']):
    for metallicity in metallicities:
        for ratio in C_O_ratios:
            # INPUT: Your path to output directory
            output_dir = 'Models/' + str(metallicity) + 'x_metallicity_' + str(ratio) + '_CO_ratio/' + \
                         str(int(min_wavelength)) + '_' + str(int(max_wavelength)) + '/'

            if len(species)==1:
                name = species[0]
            else:
                species.sort()
                name = '_'.join(species)

            # Get clear data: calculates it if doesn't already exist
            path = Path(output_dir + 'pRT_data_' + name + '.dat')
            if path.is_file():
                pass

            else:
                print(
                    "No data currently exists for this combination of species/wavelength range, calculating this atmosphere now - using default haze factor of 10")
                pRT_name = []
                for i in range(len(species)):
                    pRT_index = plot_species_lables.index(species[i])
                    pRT_name.append(pRT_names[pRT_index])

                # Calculate clear atmosphere
                calculate_atm(pRT_name, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                              metallicities=[metallicity],
                              C_O_ratios=[ratio])

            with open(path, 'r') as data:
                lines = data.readlines()
                data.close()

            wavelength_clear = []
            normalised_flux_clear = []
            for line in lines[4:]:
                v = line.split(' ')
                wavelength_clear.append(float(v[0]))
                normalised_flux_clear.append(float(v[1]))


            # Get cloudy data: calculates it if not already existing
            path = Path(output_dir + 'pRT_data_' + name + '_Cloudy.dat')
            if path.is_file():
                pass

            else:
                print("No hazy data currently exists for this combination of species/wavelength range, calculating this atmosphere now - using default haze factor of 10")
                pRT_name = []
                for i in range(len(species)):
                    pRT_index = plot_species_lables.index(species[i])
                    pRT_name.append(pRT_names[pRT_index])

                # Calculate cloudy atmosphere
                calculate_atm(pRT_name, min_wavelength=min_wavelength, max_wavelength=max_wavelength, metallicities=[metallicity],
                              C_O_ratios=[ratio], haze_factor = 10)


            with open(path,'r') as data:
                lines = data.readlines()
                data.close()

            wavelength = []
            normalised_flux = []
            for line in lines[4:]:
                v = line.split(' ')
                wavelength.append(float(v[0]))
                normalised_flux.append(float(v[1]))

            plt.figure(figsize=[15, 9])
            plt.plot(wavelength_clear, normalised_flux_clear, label='Clear model', alpha=1)
            plt.plot(wavelength, normalised_flux, label='Hazy model', alpha=1)

            plt.xlabel('Wavelength (nm)')
            plt.ylabel(r'Normalised Flux')
            plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
            plt.legend()
            plt.title('Normalised Transmission Spectrum showing clear and hazy models of ' +name+ ' at ' + str(
                metallicity) + 'x solar metallicity, ' + str(ratio)
                      + ' C/O ratio', wrap=True)
            file = Path(output_dir + 'Clear_Hazy_' + name + '.pdf')

            file.touch(exist_ok=True)
            plt.savefig(file)
            plt.clf()




### EXAMPLES:
#To generate individual spectra for all species over all 100nm wavelength chunks
species = ['H2O', 'CH4', 'CO2', 'CO', 'NH3']
species.sort()
sp          = '_'.join(i for i in species)
#for i in range(len(species)):
#    for j in range(900,2700,100):
#       generate_spectrum(min_wavelength=j, max_wavelength=j+100, orders='yes', species=[species[i]])

for j in range(900,2700,100):
   generate_spectrum(min_wavelength=j, max_wavelength=j+100, orders='yes', species=species)


# take the 100nm models and concatenate them
for met in metallicities:
    for rat in C_O_ratios:
        mod_dir = 'Models/{}x_metallicity_{}_CO_ratio/'.format(met,rat)
        # dig out the models at each wavelength
        mod_files = sorted(glob.glob(mod_dir+'*/pRT_data_{}.dat'.format(sp)))
        W_mod = []
        T_depth = []
        for ifile in mod_files:
            with open(ifile, 'r') as data:
                lines = data.readlines()
                data.close()
            for line in lines[4:]:
                v = line.split(' ')
                W_mod.append(float(v[0]))
                T_depth.append(float(v[1].split('\n')[0]))
        W_mod = np.array(W_mod)
        l = np.argsort(W_mod)
        W_mod = W_mod[l]
        T_depth = np.array(T_depth)[l]
        file = Path(mod_dir+'pRT_data_full_{}.dat'.format(sp))
        file.touch(exist_ok=True)
        with open(file, 'w') as pRT_output:
            pRT_output.write('Atmospheric Species: ')
            pRT_output.write('{}'.format(sp) + '\n')
            pRT_output.write('Metallicity: ' + str(met) + '\n')
            pRT_output.write('C/O_Ratio: ' + str(rat) + '\n')
            pRT_output.write('Wavelength[nm] Flux_variation\n')
            for i in range(len(W_mod)):
                pRT_output.writelines(str(W_mod[i]) + ' ' + str(T_depth[i]) + '\n')
            pRT_output.close
# To generate full spectra over all 100nm wavelength chunks:
#for i in range(900,2700,100):
#    generate_spectrum(min_wavelength=i, max_wavelength=i+100, orders='yes')

# Plotting the contribution spectrum: (default all species)
#for i in range(1000,2500,100):
#    contribution_spectrum(i, i+100)

#Plotting the varying metallicity spectrum: (default all species)
#for i in C_O_ratios:
    #for j in range(1000,2500,100):
        #varying_metallicity(i, j, j+100)

#Plotting the varying C/O ratio spectrum: (default all species)
#for i in metallicities:
#    for j in range(1000,2500,100):
#        varying_ratio(i, j, j+100)

#Plotting full varying metallicity spectrum for each species individually
#LS = ['CO', 'CH4', 'CO2', 'NH3', 'H2O']
#for i in range(len(LS)):
#    varying_metallicity(1.5, 1000, 2500, [LS[i]])

#for i in range(len(LS)):
#    for metallicity in metallicities:
#        varying_ratio(metallicity, 1000, 2500, [LS[i]])

# Plotting spectrum comparing clear and hazy models:
#clear_cloudy(1700,1800, species=['CH4', 'CO'])

# MAKE SURE NOTHING IS GOING TO BE RUN IN CALCULATE_ATMOSPHERE.PY AS OTHERWISE IT WILL RUN THAT HERE WHEN RUNNING THIS FILE
