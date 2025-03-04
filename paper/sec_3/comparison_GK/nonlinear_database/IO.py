import os
import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt
import source.ae as ae


matts_path = '/Users/rjjm/Library/CloudStorage/ProtonDrive-ralf.mackenbach@proton.me-folder/ITG_data/matts_data'
AE_path    = "/Users/rjjm/Library/CloudStorage/ProtonDrive-ralf.mackenbach@proton.me-folder/ITG_data/AE_data/"



def load_data(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def AE_dictionary_matrix(data, idx):
    # get the data, 1D array
    matrix = data['matrix'][idx,:]
    # get the z functions
    z_functions = data['z_functions']
    # get nl
    nl = data['nl']
    # loop over the z functions
    dict = {}
    for i, z in enumerate(z_functions):
        dict[z] = matrix[i*nl:(i+1)*nl]
    # also construct sqrt(gds2 * gds22_over_shat_squared - gds21_over_shat**2)
    dict['grad(x)Xgrad(y)'] = np.sqrt(dict['gds2'] * dict['gds22_over_shat_squared'] - dict['gds21_over_shat']**2)
    # get the name 
    tube_files = data['tube_files']
    dict['tube_name']=tube_files[idx]
    # get nfp 
    # first check if nfp is in the name
    if 'nfp' in dict['tube_name']:
        nfp = int(dict['tube_name'].split('nfp')[1][0])
    else:
        nfp = data["scalar_feature_matrix"][idx,0]
    dict['nfp'] = nfp
    return dict


def AE_dictionary_tensor(data,idx):
    # get the data, tensor shape
    tensor = data['tensor']
    # get the z functions
    z_functions = data['z_functions']
    # each [idx,:,z_idx] corresponds to z functions
    # loop over the z functions
    dict = {}
    for i, z in enumerate(z_functions):
        dict[z] = tensor[idx,:,i]
    # also construct sqrt(gds2 * gds22_over_shat_squared - gds21_over_shat**2)
    dict['grad(x)Xgrad(y)'] = np.sqrt(dict['gds2'] * dict['gds22_over_shat_squared'] - dict['gds21_over_shat']**2)
    # add name 
    dict['tube_name']=data['tube_files'][idx]
    # get nfp 
    # first check if nfp is in the name
    if 'nfp' in dict['tube_name']:
        nfp = int(dict['tube_name'].split('nfp')[1][0])
    else:
        nfp = data["scalar_feature_matrix"][idx,0]
    dict['nfp'] = nfp


    return dict
    

def AE_dictionary_hdf5(data,idx):
    # get the keys, and get the key of that index
    list_names = list(data.keys())
    name_key   = list_names[idx]
    sim = data[name_key]
    # get the main subs
    input = sim['input']
    output = sim['output']
    geom = input['geom']
    # create a dict
    dict = {}
    # loop over keys of geom
    geom_keys = list(geom.keys())
    for key in geom_keys:
        if key == 'B':
            key_dict = 'bmag'
        else:
            key_dict = key
        dict[key_dict] = np.asarray(geom[key])

    # also construct sqrt(gds2 * gds22_over_shat_squared - gds21_over_shat**2)
    dict['grad(x)Xgrad(y)'] = np.sqrt(dict['gds2'] * dict['gds22_over_shat_squared'] - dict['gds21_over_shat']**2)  # note that due to linear interpolation
                                                                                                                    # in the GX routines and factors of
                                                                                                                    # dx/psi and dy/alpha this is NOT
                                                                                                                    # exactly equal to |B|
    # add name 
    dict['tube_name'] = name_key
    # nfp is zero for tokamaks
    dict['nfp'] = 0

    return dict




def AE_dictionary(data,idx):
    # check if the key contain matrix or tensor
    if 'matrix' in data.keys():
        dict = AE_dictionary_matrix(data, idx)
    if 'tensor' in data.keys():
        dict = AE_dictionary_tensor(data, idx)
    return dict



def function_extender(dict):
    '''
    Extend the dictionary, assuming the field line is centered at a stellarator symmetric points.
    '''
    # get the keys 'bmag', 'gbdrift', 'cvdrift', 'gbdrift0_over_shat', 'gds2', 'gds21_over_shat', 'gds22_over_shat_squared'
    keys = list(dict.keys())
    # 'bmag' is even, so we can append the left-most point to the right
    dict['bmag'] = np.append(dict['bmag'], dict['bmag'][0])
    # gbdrift is also even so we append the left-most point to the right
    dict['gbdrift'] = np.append(dict['gbdrift'], dict['gbdrift'][0])
    # cvdrift is even, too.
    # not always present, so only do operation if present
    if 'cvdrift' in dict:
        dict['cvdrift'] = np.append(dict['cvdrift'], dict['cvdrift'][0])
    # gbdrift0_over_shat is odd
    dict['gbdrift0_over_shat'] = np.append(dict['gbdrift0_over_shat'], -dict['gbdrift0_over_shat'][0])
    # gds2 is even
    dict['gds2'] = np.append(dict['gds2'], dict['gds2'][0])
    # gds21_over_shat is odd
    dict['gds21_over_shat'] = np.append(dict['gds21_over_shat'], -dict['gds21_over_shat'][0])
    # gds22_over_shat_squared is even
    dict['gds22_over_shat_squared'] = np.append(dict['gds22_over_shat_squared'], dict['gds22_over_shat_squared'][0])
    # grad(x)Xgrad(y) is even
    dict['grad(x)Xgrad(y)'] = np.append(dict['grad(x)Xgrad(y)'], dict['grad(x)Xgrad(y)'][0])

    return dict



def calc_AE(data,idx_tube,Q=None,w_n=0.9,w_T=3.0,plot=False,func_ext=True,verbose=True,length_scale='rho'):
    # read data
    dict = AE_dictionary(data,idx_tube)
    if func_ext:
        dict = function_extender(dict)
    # construct relevant quantities
    B               = dict['bmag']
    grad_drift_y    = dict['gbdrift'] 
    grad_drift_x    = dict['gbdrift0_over_shat']
    # construct linspace -1,1 for arc-length coordinate
    l       = np.linspace(-1.0,1.0,len(B))
    # construct perpendicular lengthscales
    if length_scale == 'rho':
        Dx = dict['gds22_over_shat_squared']
        Dy = dict['gds2']
    # if length-scale is fixed, replace Dx and Dy with ones
    if length_scale=='fixed':
        Dx = np.ones_like(B)
        Dy = np.ones_like(B)
    w_alpha =-grad_drift_y*Dx
    w_psi   = grad_drift_x*Dy
    # now calculate the available energy array
    AE_dict = ae.calculate_AE_arr(w_T,w_n,w_alpha,w_psi)
    # do the integral to find the total
    AE_arr = AE_dict['AE']
    k_alpha_arr = AE_dict['k_alpha']
    k_psi_arr = AE_dict['k_psi']
    AE_total = np.trapezoid(Dx * Dy * AE_arr/B,l)/np.trapezoid(Dx * Dy/B,l)
    # store in dict
    result_dict = {
        'AE_arr': np.asarray(AE_arr),
        'k_alpha_arr': np.asarray(k_alpha_arr),
        'k_psi_arr': np.asarray(k_psi_arr),
        'w_alpha': np.asarray(w_alpha),
        'w_psi': np.asarray(w_psi),
        'B': np.asarray(B)
    }
    if plot:
        plot_dict(result_dict)
    # add AE total to dict
    result_dict['AE_val'] = AE_total
    # if Q is given, add it to dict
    if Q is not None:
        result_dict['Q'] = Q
    # also add w_n and w_T
    result_dict['w_n'] = w_n
    result_dict['w_T'] = w_T
    # add tube index and name
    result_dict['tube_idx'] = idx_tube
    result_dict['tube_name'] = dict['tube_name']
    # finally, add nfp
    result_dict['nfp'] = dict['nfp']

    if verbose:
        print(f'Density: {w_n:+.3f} Temperature: {w_T:+.3f} AE: {AE_total:+.3f} Tube: {idx_tube}')
         # 'Tube: ',idx_tube,' AE: ',AE)
    return result_dict


def plot_dict(dict):
    # plot the data
    n = len(dict)-1
    half = int(np.ceil(n/2))
    fig, axs = plt.subplots(half, 2, figsize=(8, 8), constrained_layout=True)
    axs = axs.flatten()
    # make x axis, from -1 to 1
    nl = len(list(dict.values())[0])
    x = np.linspace(-1, 1, nl)
    for i, (key, value) in enumerate(dict.items()):
        if isinstance(value, (np.ndarray)):
            axs[i].plot(x, value)
            axs[i].set_title(key)
            axs[i].set_xlim(-1,1)
    plt.show()



def save_to_hdf5(list,save_path,save_name):
    # save list of result_dicts to hdf5
    # structure should be tube_name from result_dict as key, with the rest of the dict as values
    # first get the tube names
    tube_names = [list[i][0]['tube_name'] for i in range(len(list))]
    # get the keys of the dicts
    keys = list[0][0].keys()
    # create a dictionary for each tube
    tube_dict = {}
    for i, tube in enumerate(tube_names):
        tube_dict[tube] = {}
        for key in keys:
            tube_dict[tube][key] = list[i][0][key]
    # save the dictionary using h5py
    # first check if the path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # make the file name
    file_name = save_path + save_name
    # open the file
    with h5py.File(file_name, 'w') as f:
        for tube in tube_dict.keys():
            f.create_group(tube)
            for key in keys:
                f[tube].create_dataset(key, data=tube_dict[tube][key])
    print('Data saved to: ',file_name)

