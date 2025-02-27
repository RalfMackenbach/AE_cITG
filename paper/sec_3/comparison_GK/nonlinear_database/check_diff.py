# check diff between high res and low res calc
import matplotlib.pyplot as plt
import numpy as np
from IO import *
import os
import pandas as pd

# enable latex rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# load data in AE_processed_data
path = 'AE_processed_data/'


# get the list of the h5 files
files = os.listdir(path)
files = [f for f in files if f.endswith('.h5')]
files = sorted(files)

# load the data using pandas read_hdf
data = []
for f in files:
    df = pd.read_hdf(path + f)
    df = df[0]
    data.append(df)

# loop over all data, splitting the data into NFP

AEs = []
Qs = []
nfps = []

for df in data:
    # now loop over all the data
    for sim in df:

        # get AE
        AE = sim['AE_val']
        # get Q
        Q = sim['Q']
        # get nfp
        try:
            nfp = int(sim['nfp'])
        except:
            print(sim['tube_name'])
        nfps.append(nfp)
        # append the AE and Q
        AEs.append(AE)
        Qs.append(Q)
            
# convert to numpy arrays
AEs = np.array(AEs)
Qs = np.array(Qs)
nfps = np.array(nfps)

# do the same in low res
path = 'AE_processed_data/low_res/'

# get the list of the h5 files
files = os.listdir(path)
files = [f for f in files if f.endswith('.h5')]
files = sorted(files)

# load the data using pandas read_hdf
data = []
for f in files:
    df = pd.read_hdf(path + f)
    df = df[0]
    data.append(df)


# loop over all data, splitting the data into NFP

AEs_low = []
Qs_low = []
nfps_low = []

for df in data:
    # now loop over all the data
    for sim in df:

        # get AE
        AE = sim['AE_val']
        # get Q
        Q = sim['Q']
        # get nfp
        try:
            nfp = int(sim['nfp'])
        except:
            print(sim['tube_name'])
        nfps_low.append(nfp)
        # append the AE and Q
        AEs_low.append(AE)
        Qs_low.append(Q)

# convert to numpy arrays
AEs_low = np.array(AEs_low)
Qs_low = np.array(Qs_low)
nfps_low = np.array(nfps_low)

# make histograms of relative and absolute differences in AE
diff = np.abs(AEs - AEs_low)
rel_diff = np.abs(AEs - AEs_low)/np.abs(AEs)

# find largest difference
idx = np.argmax(diff)
print('Largest difference: ', diff[idx])
idx = np.argmax(rel_diff)
print('Largest relative difference: ', rel_diff[idx])
print('Values at this index: ', AEs[idx], AEs_low[idx])



bin_arr = np.geomspace(1e-7,1e3,300)
plt.hist(diff, bins=bin_arr, alpha=0.5, label='abs diff')
plt.hist(rel_diff, bins=bin_arr, alpha=0.5, label='rel diff')
# set scales to log
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
