# This script is used to check the convergence of the AE calculation

# check diff between high res and low res calc
import matplotlib.pyplot as plt
import h5py
import os
import numpy as np

path_1 = '/Users/rjjm/Library/CloudStorage/ProtonDrive-ralf.mackenbach@proton.me-folder/ITG_data/AE_data/convergence_study/nom'
path_2 = '/Users/rjjm/Library/CloudStorage/ProtonDrive-ralf.mackenbach@proton.me-folder/ITG_data/AE_data/convergence_study/dbl_nquad'
path_3 = '/Users/rjjm/Library/CloudStorage/ProtonDrive-ralf.mackenbach@proton.me-folder/ITG_data/AE_data/convergence_study/hlf_tol'


# get all the hdf5 files
files = os.listdir(path_1)
files = [f for f in files if f.endswith('.hdf5')]
# sort the files
files.sort()

# same in all folders. make containers containing AE
AE_1 = []
AE_2 = []
AE_3 = []




# loop over the files
# loop over the files
for f in files:
    # load the data
    with h5py.File(path_1+'/'+f, 'r') as hf:
        # get the data
        data = hf
        # loop over the tubes
        for tube in data.keys():
            # get the data
            AE = data[tube]['AE_val'][()]
            # append to the list
            AE_1.append(AE)
    with h5py.File(path_2+'/'+f, 'r') as hf:
        # get the data
        data = hf
        # loop over the tubes
        for tube in data.keys():
            # get the data
            AE = data[tube]['AE_val'][()]
            # append to the list
            AE_2.append(AE)
    with h5py.File(path_3+'/'+f, 'r') as hf:
        # get the data
        data = hf
        # loop over the tubes
        for tube in data.keys():
            # get the data
            AE = data[tube]['AE_val'][()]
            # append to the list
            AE_3.append(AE)

# convert to numpy arrays
AE_1 = np.asarray(AE_1)
AE_2 = np.asarray(AE_2)
AE_3 = np.asarray(AE_3)

diff_1 = np.abs(AE_1 - AE_2)
diff_2 = np.abs(AE_1 - AE_3)

log_diff_1 = diff_1/np.abs(AE_1)
log_diff_2 = diff_2/np.abs(AE_1)

# make 2x2 plots with histogram of absolute and relative error
fig, ax = plt.subplots(2, 2, figsize=(12,4), constrained_layout=True)
# plot histograms
binres = 50
bins = np.geomspace(diff_1.min(), diff_1.max(), binres)
ax[0,0].hist(diff_1, bins=bins)
ax[0,0].set_xlabel(r'$A-A_{\text{dbl quad}}$')
ax[0,0].set_ylabel('Count')
ax[0,0].set_xscale('log')

bins = np.geomspace(diff_2.min(), diff_2.max(), binres)
ax[1,0].hist(diff_2, bins=bins)
ax[1,0].set_xlabel(r'$A-A_{\text{hlf tol}}$')
ax[1,0].set_ylabel('Count')
ax[1,0].set_xscale('log')

bins = np.geomspace(log_diff_1.min()*100, log_diff_1.max()*100, binres)
ax[0,1].hist(log_diff_1*100, bins=bins)
ax[0,1].set_xlabel('Relative error [%]')
ax[0,1].set_ylabel('Count')
ax[0,1].set_xscale('log')

bins = np.geomspace(log_diff_2.min()*100, log_diff_2.max()*100, binres)
ax[1,1].hist(log_diff_2*100, bins=bins)
ax[1,1].set_xlabel('Relative error [%]')
ax[1,1].set_ylabel('Count')
ax[1,1].set_xscale('log')

for ax in ax.flatten():
    ax.set_yscale('log')


# save the figure
plt.savefig('plots/convergence_comparison.png')

plt.show()