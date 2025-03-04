# plot the scatter
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py 

# enable latex rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# load data in AE_processed_data
path = '/Users/rjjm/Library/CloudStorage/ProtonDrive-ralf.mackenbach@proton.me-folder/ITG_data/AE_data'

# get the files
files = os.listdir(path)
files = [f for f in files if f.endswith('.hdf5')]
# sort the files
files.sort()

# initialize the data
AEs = []
Qs = []

# loop over the files
for f in files:
    # load the data
    with h5py.File(path+'/'+f, 'r') as hf:
        # go into main
        key = list(hf.keys())
        key = key[0]
        # get the data
        data = hf[key]
        # loop over the tubes
        for tube in data.keys():
            # get the data
            AE = data[tube]['AE_val'][()]
            Q = data[tube]['Q'][()]
            # append to the list
            AEs.append(AE)
            Qs.append(Q)

# convert to numpy arrays
AEs = np.array(AEs)
Qs = np.array(Qs)

# plot the scatter
plt.figure(figsize=(5,5))
plt.scatter(AEs,Qs,s=1)
plt.xlabel(r'$\widehat{A}$')
plt.ylabel(r'$Q$')
plt.title(r'$AE$ vs $Q$')
# log scale
plt.yscale('log')
plt.xscale('log')

plt.show()