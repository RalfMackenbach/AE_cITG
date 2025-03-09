# plot the scatter
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
# # only keep the first file
# files = [files[0]]
# initialize the data
AEs = []
Qs = []
nfps = []
w_ns = []
w_Ts = []

# loop over the files
for f in files:
    # load the data
    with h5py.File(path+'/'+f, 'r') as hf:
        # get the data
        data = hf
        # loop over the tubes with a progress bar
        for tube in tqdm.tqdm(data.keys(), desc=f"Processing {f}"):
            # get the data
            AE = data[tube]['AE_val'][()]
            Q = data[tube]['Q'][()]
            nfp = data[tube]['nfp'][()]
            w_n = data[tube]['w_n'][()]
            w_T = data[tube]['w_T'][()]
            # append to the list
            AEs.append(AE)
            Qs.append(Q)
            nfps.append(nfp)
            w_ns.append(w_n)
            w_Ts.append(w_T)

# convert to numpy arrays
AEs = np.log10(np.array(AEs))
Qs = np.log10(np.array(Qs))

# plot the scatter on a 3x3 grid.
# from top right to bottom left nfp:
# 4, 5, 6
# 3,all,7
# 2, 0, 8
fac = 3/4
fig, ax = plt.subplots(3, 3, figsize=(fac*8,fac*8), sharex=True, sharey=True, constrained_layout=True)
# make masks for the nfp values
mask_4 = np.array(nfps) == 4
mask_5 = np.array(nfps) == 5
mask_6 = np.array(nfps) == 6
mask_3 = np.array(nfps) == 3
mask_7 = np.array(nfps) == 7
mask_2 = np.array(nfps) == 2
mask_0 = np.array(nfps) == 0
mask_8 = np.array(nfps) == 8
mask_list = [mask_4, mask_5, mask_6, mask_3, mask_7, mask_2, mask_0, mask_8]

ax_4 = [0,0]
ax_5 = [0,1]
ax_6 = [0,2]
ax_3 = [1,0]
ax_7 = [1,2]
ax_2 = [2,0]
ax_0 = [2,1]
ax_8 = [2,2]
ax_list = [ax_4, ax_5, ax_6, ax_3, ax_7, ax_2, ax_0, ax_8]

c4 = 'tab:blue'
c5 = 'tab:orange'
c6 = 'tab:green'
c3 = 'tab:red'
c7 = 'tab:purple'
c2 = 'tab:brown'
c0 = 'tab:pink'
c8 = 'tab:gray'

nfp_labels = [r'$N_{\rm fp}=4$', r'$N_{\rm fp}=5$', r'$N_{\rm fp}=6$', r'$N_{\rm fp}=3$', r'$N_{\rm fp}=7$', r'$N_{\rm fp}=2$', r'$N_{\rm fp}=0$', r'$N_{\rm fp}=8$']

colors = [c4,   c5,   c6,  c3,   c7,  c2,   c0,  c8]
alphas = [0.03, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1]
# if alphas > 1, set to 1

nfp_values = [4, 5, 6, 3, 7, 2, 0, 8]

# sort the data masks, axs, colors, and labels by nfp_values
mask_list = [mask_list[i] for i in np.argsort(nfp_values)]
ax_list = [ax_list[i] for i in np.argsort(nfp_values)]
colors = [colors[i] for i in np.argsort(nfp_values)]
nfp_labels = [nfp_labels[i] for i in np.argsort(nfp_values)]
nfp_values = [nfp_values[i] for i in np.argsort(nfp_values)]

alpha_val = 1.0
marker_size = 0.5

for i, mask in enumerate(mask_list):
    N_vals = np.sum(mask)
    print(f'number of points for {nfp_labels[i]}: {N_vals}')
    ax[ax_list[i][0], ax_list[i][1]].scatter(AEs[mask], Qs[mask], c=colors[i], alpha=alphas[i], s=marker_size)
    # also add to center plot
    ax[1,1].scatter(AEs[mask], Qs[mask], c=colors[i], alpha=alphas[i], s=marker_size)

# add text to the top left
for i, mask in enumerate(mask_list):
    ax[ax_list[i][0], ax_list[i][1]].text(0.05, 0.9, nfp_labels[i], transform=ax[ax_list[i][0], ax_list[i][1]].transAxes, fontsize=12)
    

# add 'all' to the center plot
ax[1,1].text(0.05, 0.9, r'All', transform=ax[1,1].transAxes, fontsize=12)


# set lower ylim to -1
ax[1,1].set_ylim(-2, 3)

# add a 3/2 power law to all plots
lnQ = np.linspace(-1, 2, 100)
lnA = 2/3*lnQ - 1.8
for i in range(3):
    for j in range(3):
        # gray line
        ax[i,j].plot(lnA,lnQ,ls='--',color='k')

# add y label to centre-left and x label to bottom-centre
ax[1,0].set_ylabel(r'$\log_{10} Q$')
ax[2,1].set_xlabel(r'$\log_{10} A$')

plt.show()
# close allplots
plt.close('all')

# # also scatter log10(Q) against w_n and w_T

# fig, ax = plt.subplots(1, 2, figsize=(8,4), constrained_layout=True)
# # set minimal value of Q to 1e-1
# Qs_cols = np.maximum(Qs, -1)
# sc = ax[0].scatter(w_ns, w_Ts, c=Qs_cols, s=1)
# # add colorbar
# ax[0].set_xlabel(r'$\hat{\omega}_n$')
# ax[0].set_ylabel(r'$\hat{\omega}_T$')

# # get angles
# angles = np.arctan2(w_Ts, w_ns)
# # scatter angles against Q
# sc = ax[1].scatter(angles/np.pi, Qs, c=Qs_cols, s=1)
# # add line for w_T/w_n = 2/3
# x = np.linspace(-np.pi, np.pi, 100)
# y = 2/3*x
# # add colorbar
# cbar = fig.colorbar(sc, ax=ax[1], extend='min')
# cbar.set_label(r'$\log_{10}(Q)$')
# ax[1].set_xlabel(r'$\arctan(\hat{\omega}_T/\hat{\omega}_n)/\pi$')
# ax[1].set_ylabel(r'$\log_{10}(Q)$')
# # add vline at critical angle
# ax[1].axvline(x=np.arctan(2/3)/np.pi, color='k', linestyle='--')

# # show w_T/w_n = 2/3 line
# x = np.linspace(0, 10, 100)
# y = 2/3*x
# # also add w_n = 0 line
# y_0 = 0*x
# ax[0].plot(x, y, 'k--')  
# ax[0].plot(x, y_0, 'k--')
# plt.show()



# # also scatter log10(A) against w_n and w_T
# AE_cols = np.maximum(AEs, -3)
# fig, ax = plt.subplots(1, 2, figsize=(8,4), constrained_layout=True)
# sc = ax[0].scatter(w_ns, w_Ts, c=AE_cols, s=1)
# # add colorbar
# ax[0].set_xlabel(r'$\hat{\omega}_n$')
# ax[0].set_ylabel(r'$\hat{\omega}_T$')

# # get angles
# angles = np.arctan2(w_Ts, w_ns)
# # scatter angles against Q
# sc = ax[1].scatter(angles/np.pi, AEs, c=AE_cols, s=1)
# # add line for w_T/w_n = 2/3
# x = np.linspace(-np.pi, np.pi, 100)
# y = 2/3*x
# # add colorbar
# cbar = fig.colorbar(sc, ax=ax[1], extend='min')
# cbar.set_label(r'$\log_{10}(A)$')
# ax[1].set_xlabel(r'$\arctan(\hat{\omega}_T/\hat{\omega}_n)/\pi$')
# ax[1].set_ylabel(r'$\log_{10}(A)$')
# # add vline at critical angle
# ax[1].axvline(x=np.arctan(2/3)/np.pi, color='k', linestyle='--')

# # show w_T/w_n = 2/3 line
# x = np.linspace(0, 10, 100)
# y = 2/3*x
# # also add w_n = 0 line
# y_0 = 0*x
# ax[0].plot(x, y, 'k--')  
# ax[0].plot(x, y_0, 'k--')
# plt.show()