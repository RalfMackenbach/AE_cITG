# plot the scatter
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

#check if any nfp = 1 are present, or npf>8
npf1s= np.sum(nfps == 1)
nfpgt8 = np.sum(nfps > 8)
if npf1s > 0 or nfpgt8 > 0:
    print('NFP = 1: ',npf1s)
    print('NFP > 8: ',nfpgt8)

# now make 3x3 subplots: bottom left has nfp=2, middle left has nfp=3, top left has nfp=4,
# top middle has nfp=5, top right has nfp=6, middle right has nfp=7, bottom right has nfp=8
# bottom centre has nfp = 0 (tokamak)
# centre plot combines all nfp
fig, axs = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True, sharex=True, sharey=True)

#color list for different nfp
colors = ['k' ,None,'r', 'g', 'b', 'c', 'm', 'y', 'orange']

# loop over all the data
alpha_plot = 0.1
size = 0.5
marker = '.'

nfp_idc = [[2,1],None,[2,0],[1,0],[0,0],[0,1],[0,2],[1,2],[2,2]]

for i in range(9):
    mask = nfps == i
    if nfp_idc[i] is not None:
        axs[nfp_idc[i][0],nfp_idc[i][1]].scatter(AEs[mask], Qs[mask], color=colors[i], alpha=alpha_plot, s=size, marker=marker)


# plot in the centre with the colors of the nfp
for i in range(9):
    mask = nfps == i
    axs[1, 1].scatter(AEs[mask], Qs[mask], color=colors[i], alpha=alpha_plot, s=size, marker=marker)

# set the labels manually with alpha=1.0
for i in range(9):
    if nfp_idc[i] is not None:
        axs[nfp_idc[i][0],nfp_idc[i][1]].scatter([], [], color=colors[i], label=r'$N_{\rm fp} = $' + str(i), alpha=1.0, s=10.0, marker=marker)

# make log-log plots
for i in range(3):
    for j in range(3):
        axs[i, j].set_xscale('log')
        axs[i, j].set_yscale('log')
        axs[i, j].grid()

# legend for all plots but the centre plot
for i in range(9):
    if nfp_idc[i] is not None:
        axs[nfp_idc[i][0],nfp_idc[i][1]].legend()

# centre plot legend N_{\rm fp} = all
axs[1,1].scatter([], [], color='k', label=r'$N_{\rm fp} = $' + 'all', alpha=0.0, s=10.0)
axs[1,1].legend()

# make label for bottom centre and left centre plots
axs[2, 1].set_xlabel(r'$\widehat{A}$')
axs[1, 0].set_ylabel(r'$Q$')

# set lower limit y to 0.1
plt.xlim(1e-3)
plt.ylim(0.01)

# save the figure
plt.savefig('AE_vs_Q.png', dpi=1000)

plt.show()