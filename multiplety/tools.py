import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

from multiplety.atomic_calculation import rcn,rcn2,rcg,racer,atomic_calculation_summary
from multiplety.atomic_calculation import save_atomic_calculation

from multiplety.spectrum_simulator import process_xas, process_rixs, boltz_dist, get_ground_states


"""
    Tools to calculate RIXS and XAS spectra using the functions in
atomic_calculation and spectrum_simulator.
"""
  
def generate_polarization(xray_angle,scattering_angle):
    
    """ Generates the polarization inputs. This assumes that the 'c' axis of a
    octahedra symmmetry is along the x-ray at xray_angle = 90, and that xray_angle
    rotates the sample around the 'b' axis.

    Parameters
    -----------
    xray_angle: float
        Angle between x-ray and sample surface in degrees.
    
    scattering_angle: float
        2theta in degrees.

    Returns
    -----------
    absorption_pol: list
        List with intensity factors for the absorption process.
    emitted_pol: list
        List with intensity factors for the emission process.
    """

    thout = 90 - scattering_angle + xray_angle
    
    #rixs.pol_inc = [1j/np.sqrt(2),0,1j/np.sqrt(2)]
    absorption_pol = [np.sin(xray_angle*np.pi/180)/np.sqrt(2),
                      np.cos(xray_angle*np.pi/180),
                      np.sin(xray_angle*np.pi/180)/np.sqrt(2)]
        
    emission_pol = [(np.cos(thout*np.pi/180)+1)/2.,
                     -1.*np.sin(thout*np.pi/180)/np.sqrt(2),
                     (np.cos(thout*np.pi/180)-1)/2.]
        
    return absorption_pol,emission_pol

def instr_broad(calculation, incident_resolution = 0.1,
                emitted_resolution = 0.1, calc = 'rixs'):
    
    """
    Adds a gaussian broadening to the data.
    
    Parameters
    -----------
    calculation: pandas dataframe
        Resulting dataframe from process_rixs or process_xas.
    
    incident_resolution: float
        Energy resolution of incident x-rays in eV.
    
    emitted_resolution: float
        Energy resolution of emitted x-rays in eV.
        
    calc: string
        Options are 'rixs' and 'xas', selects the type of calculation.

    Returns
    -----------
    broadened_calc: pandas dataframe
        Broadened calculation in the same format as the 'calculation' input.
    """
    
    broadened_calc = calculation.copy()
    
    if calc == 'rixs':
        # Adding Gaussian broadening
        if incident_resolution != 0.0 and emitted_resolution != 0.0:
            if incident_resolution < 0.001:
                incident_resolution = 0.001
            if emitted_resolution < 0.001:
                emitted_resolution = 0.001
                    
            mid_in = (np.max(calculation.columns.values)+np.min(calculation.columns.values))/2
            mid_out = (np.max(calculation.index.values)+np.min(calculation.index.values))/2
            
            x,y = np.meshgrid(calculation.columns.values,calculation.index.values)
            gauss = np.exp(-((mid_in-x)**2/2/incident_resolution**2 + (mid_out-y)**2/2/emitted_resolution**2)) 
            gauss *= 1/2.0/np.pi/incident_resolution/emitted_resolution
        
            broadened = signal.fftconvolve(calculation.as_matrix(), gauss, mode = 'same')

            broadened_calc = pd.DataFrame(broadened,index=calculation.index.values,columns=calculation.columns.values)
        else:
            print('Incident and emitted x-ray energy resolution is set as zero!')
            
            
    if calc == 'xas':

        # Adding Gaussian broadening
        if incident_resolution > 0.0:
                    
            mid_in = (np.max(calculation.index.values)+np.min(calculation.index.values))/2
            gauss = np.exp(-(mid_in-calculation.index.values)**2/2/incident_resolution**2)
            gauss *= 1/np.sqrt(2*np.pi)/incident_resolution

            broadened_calc['right'] = signal.fftconvolve(calculation['right'], gauss, mode = 'same')
            broadened_calc['left'] = signal.fftconvolve(calculation['left'], gauss, mode = 'same')
            broadened_calc['parallel'] = signal.fftconvolve(calculation['parallel'], gauss, mode = 'same')

        else:
            print('Incident x-ray energy resolution is set as zero!')
        
    return broadened_calc
        
       
def calculate_xas(atomic_number,atom_label,gs_config,es_config,
                  crystal_field,symmetry,magnetic,fdd,fpd,gpd,coreso,valso,
                  dq10,d1,d3,mag_energy,save_folder,states_index,
                  lifetime_width,incident_resolution,ein_min = 1E10,
                  ein_max=-1E10,ein_step=0.1,temperature = 1,
                  verbose = False, skip_calc = False):
    
    """ Batch to calculate XAS. skip_calc can be used to avoid re-running all
    calculations, moving directly to XAS process.
    
    Parameters
    -----------
    atomic_number: int
        Atomic number
        
    atom_label: string
        Label for this ion (usually I use the atom plus its valence,
    for instance: Ni2+)
    
    gs_config: string
        Ground state configuration.
    
    es_config: string
        Excited state configuration.
    
    crystal_field: boolean
        Determines if crystal field will be used.
        
    symmetry: string
        Contains the symmetry of the crystal field to be applied.
        
    magnetic: boolean
        Determines if magnetic exchange will be used.
    
    fdd: int
        Reduction of the atomic Fdd slater integral in percent.

    fpd: int
        Reduction of the atomic Fpd slater integral in percent.  
    
    Gpd: int
        Reduction of the atomic Gpd slater integral in percent.

    coreso: int
        Reduction of the atomic core spin orbit coupling in percent.

    valso: int
        Reduction of the atomic core spin orbit coupling in percent.
    
    dq10: float
        Crystal field 10Dq energy.
    
    d1: float
        Difference in energy between yz/zx and xy d orbitals (in eV).
    
    d3: float
        Difference in energy between 3z2-r2 and x2-y2 d orbitals (in eV).
        
    magnetic: boolean
        Determines if magnetic exchange will be used.
    
    mag_energy: float
        Magnetic exchange energy (in eV).
        
    save_folder: string
        Folder where files will be saved.
        
    states_index: list
        Index of the states used in the calculation, the ground state is 1.
        
    lifetime_width: float
        Core hole lifetime broadening.
        
    incident_resolution: float
        Energy resolution of incident x-rays in eV.
    
    temperature: float
        Temperature to be used to calculate the weight.
    
    ein_min: float
        Minimum energy to be used in the calculation. If not specified, it
    will use the minimum final state energy minus 2 eV.

    ein_max: float
        Maximum energy to be used in the calculation. If not specified, it
    will use the maximum final state energy plus 2 eV. 
    
    ein_step: float
        Energy step to be used in the calculation.
    
    verbose: boolean
        Will print the crystal field energies if true.
        
    skip_calc: boolean
        If false it will skip the atomic calculation and use the files in
    save_folder.
    
    Returns
    -----------
    xas: pandas dataframe
        Contains the calculated XAS. Columns are labeled after the x-ray
    polarizations and index is the incident energies.
    """
    
    if skip_calc == False:
        
        bra_config = es_config
        ket_config = gs_config
        
        rcn(atomic_number,atom_label,bra_config,ket_config)
        rcn2()
        rcg(bra_config,ket_config,crystal_field,symmetry,magnetic,fdd,fpd,gpd,coreso,valso)
        racer(bra_config,ket_config,crystal_field,symmetry,dq10,d1,d3,
              magnetic,mag_energy,verbose = verbose)
        
        atomic_calculation_summary(atom_label,ket_config,crystal_field,symmetry,
                                   dq10,d1,d3,magnetic,mag_energy,verbose = verbose)
        
        save_atomic_calculation(save_folder,rixs_calculation = False, file_label = '')
    
    #XAS
    xas = process_xas(save_folder,states_index,lifetime_width,ein_min=ein_min,
                      ein_max=ein_max,ein_step=ein_step,verbose=verbose,
                      temperature=temperature)
    xas = instr_broad(xas,incident_resolution=incident_resolution,calc = 'xas')
    
    return xas

def calculate_rixs(atomic_number,atom_label,gs_config,es_config,
                   crystal_field,symmetry,magnetic,fdd,fpd,gpd,coreso,valso,
                   dq10,d1,d3,mag_energy,save_folder,
                   states_index,absorption_pol,emission_pol,
                   final_state_lifetime, intermediate_state_lifetime,
                   incident_resolution,emitted_resolution,temperature=1.,
                   ein_min = 1E10, ein_max = -1E10, ein_step = 0.1,
                   eloss_min = 1E10, eloss_max = -1E10, eloss_step = 0.1,
                   verbose = False, skip_calc = False):
   
    """ Batch to calculate RIXS. skip_calc can be used to avoid re-running all
    calculations, moving directly to rixs process.
    
    Parameters
    -----------
    atomic_number: int
        Atomic number
        
    atom_label: string
        Label for this ion (usually I use the atom plus its valence,
    for instance: Ni2+)
    
    
    gs_config: string
        Ground state configuration.
    
    es_config: string
        Excited state configuration.
    
    crystal_field: boolean
        Determines if crystal field will be used.
        
    symmetry: string
        Contains the symmetry of the crystal field to be applied.
        
    magnetic: boolean
        Determines if magnetic exchange will be used.
    
    fdd: int
        Reduction of the atomic Fdd slater integral in percent.

    fpd: int
        Reduction of the atomic Fpd slater integral in percent.  
    
    Gpd: int
        Reduction of the atomic Gpd slater integral in percent.

    coreso: int
        Reduction of the atomic core spin orbit coupling in percent.

    valso: int
        Reduction of the atomic core spin orbit coupling in percent.
    
    dq10: float
        Crystal field 10Dq energy.
    
    d1: float
        Difference in energy between yz/zx and xy d orbitals (in eV).
    
    d3: float
        Difference in energy between 3z2-r2 and x2-y2 d orbitals (in eV).
        
    magnetic: boolean
        Determines if magnetic exchange will be used.
    
    mag_energy: float
        Magnetic exchange energy (in eV).
        
    save_folder: string
        Folder where files will be saved.
        
    states_index: list
        Index of the states used in the calculation, the ground state is 1.
        
    absorption_pol: list
        Contains the incident intensity factors for the experimental geometry 
    used. Uses the [RIGHT,PARALLEL,LEFT] convention.
        
    emission_pol: list
        Contains the emitted intensity factors for the experimental geometry 
    used. Uses the [RIGHT,PARALLEL,LEFT] convention.
    
    final_state_lifetime: float
        Lifetime width of the final state (without core-hole).
    
    intermediate_state_lifetime: float
        Lifetime width of the intermediate state (with core-hole).
     
    incident_resolution: float
        Energy resolution of incident x-rays in eV.
    
    emitted_resolution: float
        Energy resolution of emitted x-rays in eV.
        
    temperature: float
        Temperature to be used to calculate the weight.
    
    eloss_min: float
        Minimum energy loss to be used in the calculation. If not specified, it
    will use the minimum final state energy minus 2 eV.

    eloss_max: float
        Maximum energy loss to be used in the calculation. If not specified, it
    will use the maximum final state energy plus 2 eV. 
    
    eloss_step: float
        Energy loss step to be used in the calculation.
    
    ein_min: float
        Minimum incident energy to be used in the calculation. If not specified,
    it will use the minimum final state energy minus 2 eV.

    ein_max: float
        Maximum incident energy to be used in the calculation. If not specified,
    it will use the maximum final state energy plus 2 eV. 
    
    ein_step: float
        Incident energy step to be used in the calculation.
    
    verbose: boolean
        Will print the crystal field energies if true.
        
    skip_calc: boolean
        If false it will skip the atomic calculation and use the files in
    save_folder.
    
    Returns
    -----------
    rixs: pandas dataframe
        Contains calculated RIXS. The incident energy is in rixs.columns.values,
    and energy loss is in rixs.index.values.
    """
    
    if skip_calc == False:
                
        #Absorption process
        
        bra_config = es_config
        ket_config = gs_config
        
        rcn(atomic_number,atom_label,bra_config,ket_config)
        rcn2()
        rcg(bra_config,ket_config,crystal_field,symmetry,magnetic,fdd,fpd,gpd,coreso,valso)
        racer(bra_config,ket_config,crystal_field,symmetry,dq10,d1,d3,
              magnetic,mag_energy,verbose = verbose)
        atomic_calculation_summary(atom_label,ket_config,crystal_field,symmetry,
                                   dq10,d1,d3,magnetic,mag_energy,verbose=verbose)
        save_atomic_calculation(save_folder,rixs_calculation = True, file_label = 'abs_')
        
        
        #Emission process
        
        bra_config = gs_config
        ket_config = es_config
        
        rcn(atomic_number,atom_label,bra_config,ket_config)
        rcn2()
        rcg(bra_config,ket_config,crystal_field,symmetry,magnetic,fdd,fpd,gpd,coreso,valso)
        racer(bra_config,ket_config,crystal_field,symmetry,dq10,d1,d3,
              magnetic,mag_energy,verbose = verbose)
        atomic_calculation_summary(atom_label,ket_config,crystal_field,symmetry,
                                   dq10,d1,d3,magnetic,mag_energy,verbose=verbose)
        save_atomic_calculation(save_folder,rixs_calculation = True, file_label = 'emi_')
        
    #RIXS

    rixs = process_rixs(save_folder,states_index,temperature,absorption_pol,emission_pol,
                        final_state_lifetime, intermediate_state_lifetime,
                        ein_min = ein_min, ein_max = ein_max, ein_step = ein_step,
                        eloss_min = eloss_min, eloss_max = eloss_max, eloss_step = eloss_step,
                        verbose = verbose)
    
    rixs = instr_broad(rixs,incident_resolution = incident_resolution,
                       emitted_resolution = emitted_resolution, calc = 'rixs')
    
    return rixs
    

def plot_rixs(rixs, ax, cmap = 'jet', zmin = None, zmax = None):
    
    """ Plot the RIXS energy map calculated.
    
    
    Parameters
    -----------
    rixs: pandas dataframe
        Contains calculated RIXS. The incident energy is in rixs.columns.values,
    and energy loss is in rixs.index.values.
    
    ax: matplotlib axis object
        Axis to plot.
    
    cmap: matplotlib colormap
        One of the matplotlib colormap options.
        
    zmin: float
        Minimum intensity.
    
    zmax: float
        Maximum intensity.

    Returns
    -----------
    rixs_plot: matplotlib artist
    """

    eloss_min = rixs.index.min()
    eloss_max = rixs.index.max()
    ein_min = rixs.columns.min()
    ein_max = rixs.columns.max()
        
    if zmin is None:
        zmin = 0
    if zmax is None:
        zmax = rixs.as_matrix().max()    
    
    ein, eloss = np.meshgrid(rixs.columns.values,rixs.index.values)
    
    plt.sca(ax)
    
    rixs_plot = plt.pcolor(ein, eloss, rixs.as_matrix(), cmap = cmap,
                           vmin = zmin, vmax = zmax)
    plt.xlim(ein_min, ein_max)
    plt.ylim(eloss_min, eloss_max)
    
    plt.xlabel('Incident energy (eV)', fontsize = 12)
    plt.ylabel('Energy loss (eV)', fontsize = 12)
    plt.colorbar()
    
    return rixs_plot
    
    
def plot_xas(xas, ax, kind = 'linear'):
    
    """ Plot XAS.
    
    Parameters
    -----------
    xas: pandas dataframe
        Contains the calculated XAS. Columns are labeled after the x-ray
    polarizations and index is the incident energies.
    
    ax: matplotlib axis object
        Axis to plot.
    
    kind: string
        Selects what geometry to plot. Options are 'linear' (plots e||c and 
    e||ab), or 'circular' (plots Parallel, Right, and Left)

    Returns
    -----------
    xas_plot: matplotlib artist
    """
    
    ein_min = xas.index.min()
    ein_max = xas.index.max()
    ymax = xas.as_matrix().max()*1.05
            
    plt.sca(ax)
    
    if kind == 'linear':
        xas_plot = [plt.plot(xas.index.values, xas['parallel'], label = 'e||c')]
        xas_plot.append(plt.plot(xas.index.values, (xas['right']+xas['left'])/2., label = 'e||ab'))
        
    elif kind == 'circular':
        xas_plot = [plt.plot(xas.index.values, xas['parallel'], label = 'Parallel')]
        xas_plot.append(plt.plot(xas.index.values, xas['right'], label = 'Right'))
        xas_plot.append(plt.plot(xas.index.values, xas['left'], label = 'Left'))
        
    else:
        print('kind variable not recognized!')
        print("Acceptable inputs: 'linear', 'circular'.")
        
    plt.xlabel('Energy (eV)', fontsize = 12)
    plt.ylabel('Intensity (arb. units)', fontsize = 12)
    plt.xlim(ein_min, ein_max)
    plt.ylim(0, ymax)
    
    plt.legend()
    
    return xas_plot
                          
def save_rixs(rixs,atomic_number,atom_label,gs_config,es_config,
              crystal_field,symmetry,magnetic,fdd,fpd,gpd,coreso,valso,
              dq10,d1,d3,mag_energy,save_folder,
              states_index,absorption_pol,emission_pol,
              final_state_lifetime, intermediate_state_lifetime,
              incident_resolution,emitted_resolution,temperature,
              fname = 'rixs_matrix.dat'):
    
    """ Save the RIXS matrix and calculation details to file fname in the 
    calculation folder. First line has the incident energy and first 
    column the energy loss.
    
    Parameters
    -----------
    rixs: pandas dataframe
        Contains calculated RIXS. The incident energy is in rixs.columns.values,
    and energy loss is in rixs.index.values.
    
    atomic_number: int
        Atomic number
        
    atom_label: string
        Label for this ion (usually I use the atom plus its valence,
    for instance: Ni2+)
    
    
    gs_config: string
        Ground state configuration.
    
    es_config: string
        Excited state configuration.
    
    crystal_field: boolean
        Determines if crystal field will be used.
        
    symmetry: string
        Contains the symmetry of the crystal field to be applied.
        
    magnetic: boolean
        Determines if magnetic exchange will be used.
    
    fdd: int
        Reduction of the atomic Fdd slater integral in percent.

    fpd: int
        Reduction of the atomic Fpd slater integral in percent.  
    
    Gpd: int
        Reduction of the atomic Gpd slater integral in percent.

    coreso: int
        Reduction of the atomic core spin orbit coupling in percent.

    valso: int
        Reduction of the atomic core spin orbit coupling in percent.
    
    dq10: float
        Crystal field 10Dq energy.
    
    d1: float
        Difference in energy between yz/zx and xy d orbitals (in eV).
    
    d3: float
        Difference in energy between 3z2-r2 and x2-y2 d orbitals (in eV).
        
    magnetic: boolean
        Determines if magnetic exchange will be used.
    
    mag_energy: float
        Magnetic exchange energy (in eV).
        
    save_folder: string
        Folder where files will be saved.
        
    states_index: list
        Index of the states used in the calculation, the ground state is 1.
        
    absorption_pol: list
        Contains the incident intensity factors for the experimental geometry 
    used. Uses the [RIGHT,PARALLEL,LEFT] convention.
        
    emission_pol: list
        Contains the emitted intensity factors for the experimental geometry 
    used. Uses the [RIGHT,PARALLEL,LEFT] convention.
    
    final_state_lifetime: float
        Lifetime width of the final state (without core-hole).
    
    intermediate_state_lifetime: float
        Lifetime width of the intermediate state (with core-hole).
     
    incident_resolution: float
        Energy resolution of incident x-rays in eV.
    
    emitted_resolution: float
        Energy resolution of emitted x-rays in eV.
        
    temperature: float
        Temperature to be used to calculate the weight.
    
    fname: string, default 'rixs_matrix.dat'
        Name of the file to be saved in 'save_folder'.
    
    Returns
    -----------
    None
    
    Files
    -----------
    'fname'
        Contains the calculated RIXS. The header contains the information about
    the calculation.
    """
    
    header = '# {:s}, Z = {:d}\n'.format(atom_label, atomic_number)
    header = header + '# Ground state: {:s}, Excited state: {:s}\n'.format(gs_config, es_config)
    header = header + '# Slater-Condon integrals reduction (100% means no reduction): Fdd = {:d}%, Fpd = {:d}%, Gpd = {:d}%\n'.format(fdd, fpd, gpd)
    header = header + '# Spin-Orbit coupling reduction: Core = {:d}%, Valence = {:d}%\n'.format(coreso,valso)
    
    header += '#\n'
    if crystal_field is True:    
        header = header + '# {:s} crystal field symmetry\n'.format(symmetry)
        header = header + '# 10dq =  {:0.3f} eV'.format(dq10)
        if symmetry != 'Oh':
            header = header + ', d1 =  {:0.3f} eV, d3 = {:0.3f} eV\n'.format(d1, d3)
            if magnetic is True:
                header = header + '# Magnetic exchange = {:0.3f} eV\n'.format(mag_energy)
        else:
            header = header + '\n'
    
    header += '#\n'        
    header = header + '# Initial states used and their weight (T = {:0.1f} K)\n'.format(temperature)
    header += '# State Energy Weight\n'
    states = get_ground_states(save_folder,states_index,prefix='abs_')
    states['boltz'] = boltz_dist(states['energy'],temperature,verbose = False)
    for state,energy,boltz in zip(states['label'],states['energy'],states['boltz']):
        header = header + '# {} {:0.3f} ({:0.1f}%)\n'.format(state,energy,boltz*100.0)
        
    header += '#\n'    
    header = header + '# Incident x-ray polarization = '
    for i in absorption_pol:
        header = header + '{:.3f}, '.format(i)
    header = header[:-2] + '\n'
    header = header + '# Emitted x-ray polarization = '
    for i in emission_pol:
        header = header + '{:.3f}, '.format(i)
    header = header[:-2] + '\n'
    header = header + '# Lifetime broadening (lorentzian) - Intermediate state = {:0.2f}, Final state = {:0.2f}\n'.format(intermediate_state_lifetime,final_state_lifetime)
    header = header + '# Intrumental broadening (gaussian) - Incident x-ray = {:0.2f}, Emitted x-ray = {:0.2f}\n'.format(incident_resolution,emitted_resolution)
    
    header += '#\n'
    header = header + '# Incident energies at the first line and energy loss at first column\n'

    file = open(save_folder + fname,'w')
    file.write(header + rixs.to_csv()[1:])
    file.close()
    
    
def save_xas(xas,atomic_number,atom_label,gs_config,es_config,
             crystal_field,symmetry,magnetic,fdd,fpd,gpd,coreso,valso,
             dq10,d1,d3,mag_energy,save_folder,states_index,
             lifetime_width,incident_resolution,temperature,fname = 'xas.dat'):
    """
    Save the XAS matrix and calculation details to file fname in the 
    calculation folder. It has four columns: energy, parallel, right, left,
    the last 3 are each x-ray polarization.
    
    
    Parameters
    -----------
    xas: pandas dataframe
        Contains the calculated XAS. Columns are labeled after the x-ray
    polarizations and index is the incident energies.
    
    atomic_number: int
        Atomic number
        
    atom_label: string
        Label for this ion (usually I use the atom plus its valence,
    for instance: Ni2+)
    
    gs_config: string
        Ground state configuration.
    
    es_config: string
        Excited state configuration.
    
    crystal_field: boolean
        Determines if crystal field will be used.
        
    symmetry: string
        Contains the symmetry of the crystal field to be applied.
        
    magnetic: boolean
        Determines if magnetic exchange will be used.
    
    fdd: int
        Reduction of the atomic Fdd slater integral in percent.

    fpd: int
        Reduction of the atomic Fpd slater integral in percent.  
    
    Gpd: int
        Reduction of the atomic Gpd slater integral in percent.

    coreso: int
        Reduction of the atomic core spin orbit coupling in percent.

    valso: int
        Reduction of the atomic core spin orbit coupling in percent.
    
    dq10: float
        Crystal field 10Dq energy.
    
    d1: float
        Difference in energy between yz/zx and xy d orbitals (in eV).
    
    d3: float
        Difference in energy between 3z2-r2 and x2-y2 d orbitals (in eV).
        
    magnetic: boolean
        Determines if magnetic exchange will be used.
    
    mag_energy: float
        Magnetic exchange energy (in eV).
        
    save_folder: string
        Folder where files will be saved.
        
    states_index: list
        Index of the states used in the calculation, the ground state is 1.
        
    lifetime_width: float
        Core hole lifetime broadening.
        
    incident_resolution: float
        Energy resolution of incident x-rays in eV.
    
    temperature: float
        Temperature to be used to calculate the weight.
    
    fname: string, default 'xas.dat'
        Name of the file to be saved in 'save_folder'.
    
    Returns
    -----------
    None
    
    Files
    -----------
    'fname'
        Contains the calculated XAS. The header contains the information about
    the calculation.
    """
    
    header = '# {:s}, Z = {:d}\n'.format(atom_label, atomic_number)
    header = header + '# Ground state: {:s}, Excited state: {:s}\n'.format(gs_config, es_config)
    header = header + '# Slater-Condon integrals reduction (100% means no reduction): Fdd = {:d}%, Fpd = {:d}%, Gpd = {:d}%\n'.format(fdd, fpd, gpd)
    header = header + '# Spin-Orbit coupling reduction: Core - {:d}%, Valence - {:d}%\n'.format(coreso,valso)
    
    header += '#\n'
    if crystal_field is True:    
        header = header + '# {:s} crystal field symmetry\n'.format(symmetry)
        header = header + '# 10dq =  {:0.3f} eV'.format(dq10)
        if symmetry != 'Oh':
            header = header + ', d1 =  {:0.3f} eV, d3 = {:0.3f} eV\n'.format(d1, d3)
            if magnetic is True:
                header = header + '# Magnetic exchange = {:0.3f} eV\n'.format(mag_energy)
        else:
            header = header + '\n'
            
    header += '#\n'      
    header = header + '# Initial states used and their weight (T = {:0.1f} K)\n'.format(temperature)
    header += '# State Energy Weight\n'
    states = get_ground_states(save_folder,states_index)
    states['boltz'] = boltz_dist(states['energy'],temperature,verbose = False)
    for state,energy,boltz in zip(states['label'],states['energy'],states['boltz']):
        header = header + '# {} {:0.3f} ({:0.1f}%)\n'.format(state,energy,boltz*100.0)

    header += '#\n'
    header = header + '# Lifetime broadening (lorentzian) = {:0.2f}\n'.format(lifetime_width)
    header = header + '# Instrumental broadening (gaussian) = {:0.2f}\n'.format(incident_resolution)
    
    header += '#\n'
    file = open(save_folder + fname,'w')
    file.write(header + xas.to_csv()[1:])
    file.close()
    
def load_xas(save_folder, fname):
    
    """ Load XAS data saved with save_xas.
    
    Parameters
    -----------
    save_folder: string
        Folder that the calculation is saved.
    
    fname: string
        Name of the saved file.
    
    Returns
    -----------
    xas: pandas dataframe
        Contains the calculated XAS. Columns are labeled after the x-ray
    polarizations and index is the incident energies.
    """
    
    return pd.read_csv(save_folder+fname,comment='#',dtype=np.float64)

def load_rixs(save_folder, fname):
    
    """ Load RIXS data saved with save_rixs.
    
    Parameters
    -----------
    save_folder: string
        Folder that the calculation is saved.
    
    fname: string
        Name of the saved file.
    
    Returns
    -----------
    rixs: pandas dataframe
        Contains calculated RIXS. The incident energy is in rixs.columns.values,
    and energy loss is in rixs.index.values.
    """
    
    rixs = pd.read_csv(save_folder+'rixs_matrix.dat',comment='#',dtype=np.float64)
    rixs.columns = pd.to_numeric(rixs.columns)

    return rixs
    
