# Here we plot the scatter plot of AE and Q for all the tubes in the database
# with fixed gradients

import matplotlib.pyplot as plt
import numpy as np
import os
import h5py 
import tqdm
import IO

# enable latex rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# load data in AE_processed_data
path = IO.AE_path

# get the files
files = os.listdir(path)
files = [f for f in files if f.endswith('.hdf5')]
# sort the files
files.sort()
# remove RANDOM files (capitalized or not)
files = [f for f in files if 'random' not in f.lower()]
# initialize the data
AEs = []
Qs = []
nfps = []
w_ns = []
w_Ts = []
quasr_mask = []
tube_names = []
B = []

mirror_ratios = []
Qs = []
AEs = []

path_load = '/Users/rjjm/Documents/GitHub/AE_cITG/paper/sec_3/comparison_GK/nonlinear_database/data_processed/fixed/'
# check if the data is already saved
if False:
    a = 0
else:
    # loop over the files
    for f in files:
        # load the data
        with h5py.File(path+'/'+f, 'r') as hf:
            # get the data
            data = hf
            # loop over the tubes with a progress bar
            print(f'Number of tubes in {f}: {len(data.keys())}')
            for tube in tqdm.tqdm(data.keys(), desc=f"Processing {f}"):
                # get the data
                AE = data[tube]['AE_val'][()]
                Q = data[tube]['Q'][()]
                nfp = data[tube]['nfp'][()]
                w_n = data[tube]['w_n'][()]
                w_T = data[tube]['w_T'][()]
                name = data[tube]['tube_name'][()]
                
                # check if nfp=8, Q<0.1 and A>0.1
                if (nfp == 8 or nfp==7) and Q < 0.1 and AE > 0.1:
                    # print the name
                    print(f'Found outlier: {name}')
                    # plot magnetic field ('B', in data)
                    B = data[tube]['B'][()]
                    # plot the data
                    plt.plot(B)
                    plt.title(f'Outlier Q={Q}, AE={AE}')
                    plt.xlabel('z') 
                    plt.ylabel('B')
                    plt.savefig(f'outlier_{name}.png')
                    plt.close()
                    B_max = np.max(B)
                    B_min = np.min(B)
                    mirror_ratio = (B_max-B_min)/B_min
                    # append the data
                    mirror_ratios.append(mirror_ratio)
                    Qs.append(Q)
                    AEs.append(AE)
                elif (nfp == 8 or nfp == 7) and Q < 0.1 and AE < 0.01:
                    # print the name
                    print(f'Found outlier: {name}')
                    # plot magnetic field ('B', in data)
                    B = data[tube]['B'][()]
                    # plot the data
                    plt.plot(B)
                    plt.title(f'Outlier Q={Q}, AE={AE}')
                    plt.xlabel('z') 
                    plt.ylabel('B')
                    plt.savefig(f'outlier_{name}.png')
                    plt.close()
                    B_max = np.max(B)
                    B_min = np.min(B)
                    mirror_ratio = (B_max-B_min)/B_min
                    # append the data
                    mirror_ratios.append(mirror_ratio)
                    Qs.append(Q)
                    AEs.append(AE)

# make scatter plot of AE and Q, color according to mirror ratio
plt.figure(figsize=(8, 6))
plt.scatter(AEs, Qs, c=mirror_ratios, cmap='viridis', s=50)
plt.colorbar(label='Mirror Ratio')
plt.xlabel(r'$A$')
plt.ylabel(r'$Q$')
# save the figure
plt.savefig('scatter_AE_Q.png')
plt.show()