import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict

"""
    Calculate RIXS and XAS from atomic models.

Butler x Mulliken nomenclature

The crystal field symmetry is more often labeled using Mulliken nomenclature
(e.g. Eg, T2g), but the Racer code uses the Butler nomenclature. While the
conversion is not implemented here, the correspondence is:

Oh symmetry
Butler | Mulliken
  O    |    A1
 -O    |    A2
  1    |    T1
 -1    |    T2
  2    |    E

D4h symmetry
Butler | Mulliken
  O    |    A1
 -O    |    A2
  1    |    E
  2    |    T1
 -2    |    T2
"""

def boltz_dist(states_energy,temperature,verbose = False):
    """Calculates Boltzmann distribution for multiple ground states.

    Parameters
    -----------
    states_energy: list
        Energies of the states to be weighted.

    temperature: float
        Temperature to be used in the distribution.

    verbose: Boolean
        Priting option.

    Returns
    -----------
    boltz: list
        Boltzmann distribution.
    """

    if temperature > 0.001:
        boltz = np.exp(-(np.array(states_energy) - states_energy[0])/8.61733E-5/temperature)
        boltz /= np.sum(np.exp(-(np.array(states_energy) - states_energy[0])/8.61733E-5/temperature))
    else:
        boltz = [1.0]
        for i in range(len(states_energy)-1):
            boltz.append(0.0)

    if verbose == True:
        print('State energy (eV)\tWeight')
        for i in range(len(states_energy)):
            print('{:0.3f}\t{:0.3f}'.format(states_energy[i], boltz[i]))

    return boltz

def get_ground_states(save_folder,states_index,prefix=''):

    """ Collect the ground states that will be used in the calculations.

    Parameters
    -----------
    save_folder: string
        Folder where files will be saved.

    states_index: list
       Index of the states used in the calculation (items have to be > 0).

    prefix: string
        Prefix to the results.dat file. Used for RIXS calculations.

    Returns
    -----------
    states: dictionary
        Contain two items of the same size along axis 0.'label' has the
    symmetry label of the states. 'energy' has their energies.
    """

    states = OrderedDict({})
    states['index'] = states_index

    gstates = open(save_folder + prefix + 'results.dat', 'r').readlines()

    for line in gstates:
        if 'Ground state energy' in line:
            states['gs0'] = float(line.split()[-2])

    ind = gstates.index('Symmetry  Energy(eV)\n')
    states['label'] = []
    states['energy'] = []
    for i in states_index:
        line = gstates[ind+i]
        state,ee = line.split()
        states['label'].append(state)
        states['energy'].append(states['gs0'] + float(ee))

    return states

def collect_transition_matrix(save_folder,states_label,polarization,
                              states_energy = None, prefix=''):

    """ Collects the transition matrices calculated by RACER.

    Parameters
    -----------
    save_folder: string
        Folder where files will be saved.

    states_label: list
       Contains the symmetry label of the states to be used as given by RACER.

    polarization: list
        Contains the intensity factors for the experimental geometry
    used. Uses the [RIGHT,PARALLEL,LEFT] convention.

    Returns
    -----------
    matrix: dictionary
        Contain four items of the same size along axis 0.'label' has the
    transition labels. 'bra_energy' has the energies of the bra states.
    'ket_energy' has the energies of the ket states. 'matrix' has the transition
    matrices.
    """

    infile = open(save_folder + prefix + 'racer_out.ora', 'r').readlines()

    matrix = OrderedDict({})
    matrix['bra_energy'] = []
    matrix['ket_energy'] = []
    matrix['matrix'] = []
    matrix['label'] = []

    for line in infile:
        if 'TRANSFORMED MATRIX' in line:
            pol = 'No'
            for label in states_label:
                if line.split(')')[0].split()[-1] ==  label:
                    pol = line.split('\n')[0].split()[-1]
                    bra_state = line.split('(')[1].split()[0]
                    ket_state = label

            if pol != 'No':
                bra_energy = []
                ket_energy = []
                matrix_elements = []

                k = 0
                repeat_num = 0
                while 'TRANSFORMED MATRIX' not in infile[infile.index(line)+4+k]:
                    try:
                        first,rest = infile[infile.index(line)+4+k].split(':')

                        if 'BRA/KET' in first:
                            line_num = 0
                            repeat_num = repeat_num + 1

                            for i in rest.split():
                                ket_energy.append(float(i))

                            k = k + 1
                            first,rest = infile[infile.index(line)+4+k].split(':')

                        try:
                            if repeat_num == 1:
                                matrix_elements.append([])
                                bra_energy.append(float(first))
                                for i in rest.split():
                                    if pol == 'RIGHT':
                                        matrix_elements[line_num].append(polarization[0]*complex(float(i),0))
                                    if pol == 'LEFT':
                                        matrix_elements[line_num].append(polarization[2]*complex(float(i),0))
                                    if pol == 'PARALLEL':
                                        matrix_elements[line_num].append(polarization[1]*complex(float(i),0))
                                line_num = line_num + 1
                            else:
                                for i in rest.split():
                                    if pol == 'RIGHT':
                                        matrix_elements[line_num].append(polarization[0]*complex(float(i),0))
                                    if pol == 'LEFT':
                                        matrix_elements[line_num].append(polarization[2]*complex(float(i),0))
                                    if pol == 'PARALLEL':
                                        matrix_elements[line_num].append(polarization[1]*complex(float(i),0))
                                line_num = line_num + 1

                            k = k + 1
                        except ValueError:
                            k = k + 1
                    except ValueError:
                        k = k + 1

                    if 'TRANSFORMATION FINISHED' in infile[infile.index(line)+4+k]:
                        break

                bra_energy = np.array(bra_energy)
                ket_energy = np.array(ket_energy)
                matrix_elements = np.array(matrix_elements)

                if type(states_energy) is list:

                    index = [False for i in range(len(ket_energy))]
                    for energy in states_energy:
                        index += np.abs(ket_energy-energy) < 0.0002

                    ket_energy = ket_energy[index]
                    matrix_elements = matrix_elements[:,index]

                if len(bra_energy) != 0:
                    matrix['label'].append('{:s} {:s} {:s}'.format(bra_state, pol.lower(), ket_state))
                    matrix['bra_energy'].append(bra_energy)
                    matrix['ket_energy'].append(ket_energy)
                    matrix['matrix'].append(matrix_elements)

    return matrix

def process_xas(save_folder,states_index,lifetime_width,temperature = 1,
                ein_min=1E10,ein_max=-1E10,ein_step=0.1,verbose=False):

    """Collects the output of racer and calculates the XAS.

    Parameters
    -----------
    save_folder: string
        Folder where files will be saved.

    states_index: list
        Index of the states used in the calculation, the ground state is 1.

    lifetime_width: float
        Core hole lifetime broadening.

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
        Will print the weights if true.

    Returns
    -----------
    xas: pandas dataframe
        Contains the calculated XAS. Columns are labeled after the x-ray
    polarizations and index is the incident energies.
    """

    states = get_ground_states(save_folder,states_index)

    matrix = collect_transition_matrix(save_folder,states['label'],polarization = [1,1,1],
                                       states_energy = states['energy'])

    states['boltz'] = boltz_dist(states['energy'],temperature,verbose)

    # Apply Bolzmann factor
    for i in range(len(matrix['matrix'])):
        for j in range(matrix['matrix'][i].shape[1]):
            for energy,state,boltz in zip(states['energy'], states['label'],states['boltz']):
                if matrix['label'][i].split()[-1] == state:
                    if np.array(matrix['ket_energy'][i][j]-energy) < 0.0002:
                        matrix['matrix'][i][:,j] *= boltz

    # Finds energy limits
    if ein_max == -1E10:
        for line in matrix['bra_energy']:
            if ein_max < np.max(line):
                ein_max = np.max(line)
        ein_max += 2.0
    if ein_min == 1E10:
        for line in matrix['bra_energy']:
            if ein_min > np.min(line):
                ein_min = np.min(line)
        ein_min -= 2.0

    if verbose is True:
        print('Starting XAS calculation...')


    # Build transition matrices - it makes it easier to calculate later.
    matrix['parallel'] = np.array([])
    matrix['parallel_energy'] = np.array([])
    matrix['right'] = np.array([])
    matrix['right_energy'] = np.array([])
    matrix['left'] = np.array([])
    matrix['left_energy'] = np.array([])

    for i in range(len(matrix['label'])):
        if verbose is True:
            print(matrix['label'][i])

        pol = matrix['label'][i].split()[1]

        matrix['{}_energy'.format(pol)] = np.append(matrix['{}_energy'.format(pol)],matrix['bra_energy'][i])

        tmp = np.sum(np.abs(matrix['matrix'][i])**2,axis=1)
        matrix[pol] = np.append(matrix[pol],tmp)


    # Calculate XAS
    ein = np.linspace(ein_min, ein_max, int((ein_max - ein_min)/ein_step + 0.5))
    xas = pd.DataFrame(np.zeros((len(ein),3)),index=ein,columns=['parallel','right','left'])

    for incident_energy in xas.index.values:
        for key in xas.keys():
            xas[key][incident_energy] = np.sum(matrix[key]/((incident_energy - matrix['{}_energy'.format(key)])**2 + lifetime_width**2))

    if verbose is True:
        print('Done!')

    return xas

def cleanup_transition_matrix(matrix,polarization):
    """ Remove entries that are all zero to speed up the RIXS calculation.

    Parameters
    -----------
    matrix: dictionary
        Contains the transition matrices, created by the collect_transition_matrix
    function.

    polarization: list
        Contains the incident intensity factors for the experimental geometry
    used. Uses the [RIGHT,PARALLEL,LEFT] convention.

    Returns
    -----------
    matrix: dictionary
        Transition matrices with zero entries removed.
    """

    index = []
    for i in range(len(matrix['label'])):
        if (polarization[0] == 0) & ('right' in matrix['label'][i]):
            index.append(i)
        elif (polarization[1] == 0) & ('parallel' in matrix['label'][i]):
            index.append(i)
        elif (polarization[2] == 0) & ('left' in matrix['label'][i]):
            index.append(i)

    for i in reversed(index):
        del matrix['label'][i]
        del matrix['bra_energy'][i]
        del matrix['ket_energy'][i]
        del matrix['matrix'][i]

    return matrix

def process_rixs(save_folder,states_index,temperature,absorption_pol,emission_pol,
                 final_state_lifetime, intermediate_state_lifetime,
                 ein_min = 1E10, ein_max = -1E10, ein_step = 0.1,
                 eloss_min = 1E10, eloss_max = -1E10, eloss_step = 0.1,
                 verbose = False):

    """Collects the output of racer and calculates the RIXS.

    *** The method for collecting input and calculating the RIXS could be improved ***

    Parameters
    -----------
    save_folder: string
        Folder where files will be saved.

    states_index: list
        Index of the states used in the calculation, the ground state is 1.

    temperature: float
        Temperature to be used to calculate the weight.

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
        Will print the weights if true.

    Returns
    -----------
    rixs: pandas dataframe
        Contains calculated RIXS. The incident energy is in rixs.columns.values,
    and energy loss is in rixs.index.values.
    """


    states = get_ground_states(save_folder,states_index,prefix='abs_')
    states['boltz'] = boltz_dist(states['energy'],temperature,verbose=verbose)

    absorption = collect_transition_matrix(save_folder,states['label'],absorption_pol,
                                           states_energy = states['energy'], prefix='abs_')

    absorption = cleanup_transition_matrix(absorption,absorption_pol)

    emission_label = [label.split()[0] for label in absorption['label']]
    emission = collect_transition_matrix(save_folder,emission_label,emission_pol,
                                         prefix='emi_')


    # Apply Bolzmann factor
    for i in range(len(absorption['matrix'])):
        for j in range(absorption['matrix'][i].shape[1]):
            for energy,state,boltz in zip(states['energy'],states['label'],states['boltz']):
                if absorption['label'][i].split()[-1] == state:
                    if np.array(absorption['ket_energy'][i][j]-energy) < 0.0002:
                        absorption['matrix'][i][:,j] *= boltz

    # Construct rixs matrix
    rixs_matrix = OrderedDict({})
    rixs_matrix['label'] = []
    rixs_matrix['energy_inc'] = []
    rixs_matrix['energy_loss'] = []
    rixs_matrix['matrix'] = []

    for i in range(len(absorption['label'])):
        for j in range(len(emission['label'])):
            if absorption['label'][i].split()[0] == emission['label'][j].split()[-1]:
                for k in range(len(absorption['ket_energy'][i])):
                    emi, inc = np.meshgrid([np.abs(x - states['gs0']) for x in emission['bra_energy'][j]], absorption['bra_energy'][i]- absorption['ket_energy'][i][k])
                    rixs_matrix['energy_loss'].append(emi)
                    rixs_matrix['energy_inc'].append(inc)
                    #rixs_matrix['label'].append(absorption['label'][i] + ' {:s} {:s}'.format(emission['label'][j].split()[1],emission['label'][j].split()[2]))
                    rixs_matrix['label'].append(emission['label'][j] + ' {:s} {:s}'.format(absorption['label'][i].split()[1],absorption['label'][i].split()[2]))
                    rixs_matrix['matrix'].append(np.zeros((len(rixs_matrix['energy_inc'][-1]), len(rixs_matrix['energy_loss'][-1][0,:])), dtype = complex))
                    for l in range(len(rixs_matrix['matrix'][-1])):
                        rixs_matrix['matrix'][-1][l,:] = [x*absorption['matrix'][i][l,k] for x in emission['matrix'][j][:,l]]

    # Finds energy limits
    if ein_max == -1E10:
        for line in absorption['bra_energy']:
            if ein_max < np.max(line):
                ein_max = np.max(line)
        ein_max += 2.0
    if ein_min == 1E10:
        for line in absorption['bra_energy']:
            if ein_min > np.min(line):
                ein_min = np.min(line)
        ein_min -= 2.0

    if eloss_max == -1E10:
        for line in emission['bra_energy']:
            if eloss_max < (np.max(line) - states['gs0']):
                eloss_max = np.max(line) - states['gs0']
        eloss_max += 1.0
    if eloss_min == 1E10:
        for line in emission['bra_energy']:
            if eloss_min > (np.min(line) - states['gs0']):
                eloss_min = np.min(line) - states['gs0']
        eloss_min -= 1.0

    energy_loss = np.linspace(eloss_min , eloss_max, int((eloss_max - eloss_min)/eloss_step + 0.5))
    energy_inc = np.linspace(ein_min, ein_max, int((ein_max - ein_min)/ein_step + 0.5))

    if verbose is True:
        time0 = datetime.now()

    if verbose is True:
        print('\nStarting RIXS calculation...')
        print('Transitions used:')

        for l in rixs_matrix['label']:
           print(l)

        print('\nBuilding RIXS matrix...')

    #Finding transitions that will interfere
    rixs_interference = []
    rixs_interference_label = []
    for i in range(len(rixs_matrix['label'])):

        init = rixs_matrix['label'][i].split()[0]
        final = rixs_matrix['label'][i].split()[-1]

        if len(rixs_interference) == 0:
            rixs_interference_label.append('{:s},{:s}'.format(init,final))
            rixs_interference.append([i])
        else:

            if '{:s},{:s}'.format(init,final) in rixs_interference_label:
                rixs_interference[rixs_interference_label.index('{:s},{:s}'.format(init,final))].append(i)
            else:
                rixs_interference_label.append('{:s},{:s}'.format(init,final))
                rixs_interference.append([i])

    if verbose is True:
        print('Interference pairs')
        for i,j in zip(rixs_interference_label, rixs_interference):
            print(i,j)

        print('')

    rixs = pd.DataFrame(np.zeros((len(energy_loss),len(energy_inc))),index=energy_loss,columns=energy_inc)

    for eloss in energy_loss:
        if verbose is True:
            if np.abs(eloss - int(eloss+0.5)) < eloss_step/2:
                print('Eloss = {:0.2f} eV... '.format(eloss))
        for ein in energy_inc:
            aux = np.array([])
            for l in range(len(rixs_interference)):
                M = []
                for k in rixs_interference[l]:
                    M.append(np.sum(rixs_matrix['matrix'][k]/(ein-rixs_matrix['energy_inc'][k] + intermediate_state_lifetime*1j),axis=0))
                aux = np.append(aux, np.abs(np.sum(M,axis=0))**2*final_state_lifetime/2/np.pi/((eloss-rixs_matrix['energy_loss'][rixs_interference[l][0]][0,:])**2 + final_state_lifetime**2/4))
            rixs[ein][eloss] = np.sum(aux)

    if verbose is True:
        print('Done!')

    if verbose is True:
        timef = datetime.now()
        print('Time to create rixs matrixes: ', timef-time0)

    return rixs
