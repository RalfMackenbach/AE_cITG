import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import source.ae_ITG as ae


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
