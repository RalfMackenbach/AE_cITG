import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import source.ae_ITG as ae
import scipy.integrate as spi


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
                                                                                                                    # equal to unity
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



def calc_AE(data,idx_tube,Q=None,w_n=0.9,w_T=3.0,plot=False,func_ext=True,verbose=True):
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
    Dx = 1.0
    Dy = 1.0
    w_alpha =-grad_drift_y*Dx
    w_psi   = grad_drift_x*Dy
    # now calculate the available energy at each l
    AE_arr  = np.zeros_like(l)
    k_alpha_arr = np.zeros_like(l)
    k_psi_arr = np.zeros_like(l)
    for idx, _ in np.ndenumerate(AE_arr):
        idx = idx[0]
        AE_dict = ae.calculate_AE(w_alpha[idx],w_psi[idx],w_n,w_T)
        AE_arr[idx] = AE_dict['AE']
        k_alpha_arr[idx] = AE_dict['k_alpha']
        k_psi_arr[idx] = AE_dict['k_psi']

    # now do integral of AE_arr
    AE = spi.simpson(AE_arr/B,x=l)/spi.simpson(1/B,x=l)

    # construct dict with AE, AE_arr, k_alpha, k_psi
    # also save inputs w_alpha, w_psi, B
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
    
    # add AE_val and tube_idx to dict
    result_dict['AE_val'] = AE
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
        print('Density: ',w_n,' Temperature: ',w_T, ' AE: ',AE, ' Tube: ',idx_tube)
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
