# check diff between high res and low res calc
import matplotlib.pyplot as plt
import h5py
import os
import numpy as np

path = '/Users/rjjm/Library/CloudStorage/ProtonDrive-ralf.mackenbach@proton.me-folder/ITG_data/AE_data/convergence_study'

# find the hdf5 files
files = os.listdir(path)
files = [f for f in files if f.endswith('.hdf5')]
files.sort()
AE_high_res = []
AE_hlf_quad = []
AE_dbl_tol  = []

# first read 20240601.hdf5
file_name = path+'/20240601.hdf5'
f = h5py.File(file_name,'r')
# loop over tubes
for main in f.keys():
    for tube in f[main].keys():
        AE_high_res.append(f[main][tube]['AE_val'][()])
f.close()

# then read 20240601_hlf_nquad.hdf5
file_name = path+'/20240601_hlf_nquad.hdf5'
f = h5py.File(file_name,'r')
# loop over tubes
for main in f.keys():
    for tube in f[main].keys():
        AE_hlf_quad.append(f[main][tube]['AE_val'][()])
f.close()

# then read 20240601_dbl_tolerance.hdf5
file_name = path+'/20240601_dbl_tolerance.hdf5'
f = h5py.File(file_name,'r')
# loop over tubes
for main in f.keys():
    for tube in f[main].keys():
        AE_dbl_tol.append(f[main][tube]['AE_val'][()])
f.close()

# make histogram of the absolute difference and relative difference
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
diff1 = np.abs(np.array(AE_high_res) - np.array(AE_hlf_quad))
diff2 = np.abs(np.array(AE_high_res) - np.array(AE_dbl_tol))
ax[0,0].hist(diff1, bins=20)
ax[0,0].set_title(r'Absolute difference')
ax[0,0].set_xlabel(r'$|A - A_{\rm half \; quad}|$')
ax[0,0].set_ylabel('Counts')

rel_diff1 = (diff1 / np.array(AE_high_res)) * 100
ax[0,1].hist(rel_diff1, bins=20)
ax[0,1].set_title(r'Relative difference')
ax[0,1].set_xlabel(r'$|A - A_{\rm half \; quad}|/|A|$ (%)')
ax[0,1].set_ylabel('Counts')

ax[1,0].hist(diff2, bins=20)
ax[1,0].set_title(r'Absolute difference')
ax[1,0].set_xlabel(r'$|A - A_{\rm double \; tol}|$')
ax[1,0].set_ylabel('Counts')

rel_diff2 = (diff2 / np.array(AE_high_res)) * 100
ax[1,1].hist(rel_diff2, bins=20)
ax[1,1].set_title('Relative difference')
ax[1,1].set_xlabel(r'$|A - A_{\rm double \; tol}|/|A|$ (%)')
ax[1,1].set_ylabel('Counts')

# log y-axis
ax[0,0].set_yscale('log')
ax[0,1].set_yscale('log')
ax[1,0].set_yscale('log')
ax[1,1].set_yscale('log')

# set xlim to 0
ax[0,0].set_xlim(0, None)
ax[0,1].set_xlim(0, None)
ax[1,0].set_xlim(0, None)
ax[1,1].set_xlim(0, None)

plt.tight_layout()

# save figure
plt.savefig('plots/convergence_study.png')

plt.show()