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
files = [f for f in files if 'random' in f.lower()]
# initialize the data
AEs = []
Qs = []
nfps = []
w_ns = []
w_Ts = []
quasr_mask = []

path_load = '/Users/rjjm/Documents/GitHub/AE_cITG/paper/sec_3/comparison_GK/nonlinear_database/data_processed/random/'
# check if the data is already saved
if os.path.exists(path_load+'AEs.npy'):
    # load the data
    AEs = np.load(path_load+'AEs.npy')
    Qs = np.load(path_load+'Qs.npy')
    nfps = np.load(path_load+'nfps.npy')
    w_ns = np.load(path_load+'w_ns.npy')
    w_Ts = np.load(path_load+'w_Ts.npy')
    quasr_mask = np.load(path_load+'quasr_mask.npy')
    print('Data loaded from /data_processed/')
# else load the data
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
                # append to the list
                AEs.append(AE)
                Qs.append(Q)
                nfps.append(nfp)
                w_ns.append(w_n)
                w_Ts.append(w_T)
                # check if quasr
                name = name.decode('utf-8')
                if 'quasr' in name.lower():
                    quasr_mask.append(True)
                else:
                    quasr_mask.append(False)

    # save the data to numpy arrays
    AEs = np.array(AEs)
    Qs = np.array(Qs)
    nfps = np.array(nfps)
    w_ns = np.array(w_ns)
    w_Ts = np.array(w_Ts)
    quasr_mask = np.array(quasr_mask)
    # save the data
    np.save(path_load+'AEs.npy', AEs)
    np.save(path_load+'Qs.npy', Qs)
    np.save(path_load+'nfps.npy', nfps)
    np.save(path_load+'w_ns.npy', w_ns)
    np.save(path_load+'w_Ts.npy', w_Ts)
    np.save(path_load+'quasr_mask.npy', quasr_mask)
    print('Data saved to /data_processed/fixed/')

# convert to numpy arrays
AEs = np.log10(np.array(AEs))
Qs = np.log10(np.array(Qs))

# make masks for the nfp values
mask_4 = np.array(nfps) == 4
mask_5 = np.array(nfps) == 5
mask_6 = np.array(nfps) == 6
mask_3 = np.array(nfps) == 3
mask_7 = np.array(nfps) == 7
mask_2 = np.array(nfps) == 2
mask_0 = np.array(nfps) == 0
mask_8 = np.array(nfps) == 8

# Convert all masks to numpy arrays
mask_4 = np.array(mask_4)
mask_5 = np.array(mask_5)
mask_6 = np.array(mask_6)
mask_3 = np.array(mask_3)
mask_7 = np.array(mask_7)
mask_2 = np.array(mask_2)
mask_0 = np.array(mask_0)
mask_8 = np.array(mask_8)
mask_list = [mask_4, mask_5, mask_6, mask_3, mask_7, mask_2, mask_0, mask_8]

# make masks for names (quasr)
mask_quasr = quasr_mask

ax_4 = [0,0]
ax_5 = [0,1]
ax_6 = [0,2]
ax_3 = [1,0]
ax_7 = [1,2]
ax_2 = [2,0]
ax_0 = [2,1]
ax_8 = [2,2]
ax_list = [ax_4, ax_5, ax_6, ax_3, ax_7, ax_2, ax_0, ax_8]

nfp_labels = [r'$N_{\rm fp}=4$', r'$N_{\rm fp}=5$', r'$N_{\rm fp}=6$', r'$N_{\rm fp}=3$', r'$N_{\rm fp}=7$', r'$N_{\rm fp}=2$', r'$N_{\rm fp}=0$', r'$N_{\rm fp}=8$']

alphas = np.asarray([1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0])/3

nfp_values = [4, 5, 6, 3, 7, 2, 0, 8]

# sort the data masks, axs, colors, and labels by nfp_values
mask_list = [mask_list[i] for i in np.argsort(nfp_values)]
ax_list = [ax_list[i] for i in np.argsort(nfp_values)]
nfp_labels = [nfp_labels[i] for i in np.argsort(nfp_values)]
nfp_values = [nfp_values[i] for i in np.argsort(nfp_values)]
alpha_vals = [alphas[i] for i in np.argsort(nfp_values)]

alpha_val = 1.0
marker_size = 0.20

######################################################################################################################
######################################################################################################################
###################################################### PLOTTING ######################################################
######################################################################################################################
######################################################################################################################

# now plot the data
fac = 3/4
fig, ax = plt.subplots(3, 3, figsize=(fac*8,fac*8), sharex=True, sharey=True, constrained_layout=True)

quasr_c = 'tab:blue'
else_c = 'tab:red'

for i, mask in enumerate(mask_list):
    N_vals = np.sum(mask)
    print(f'number of points for {nfp_labels[i]}: {N_vals}')
    # plot the scatter for else
    ax[ax_list[i][0], ax_list[i][1]].scatter(AEs[mask & (~mask_quasr)], Qs[mask & (~mask_quasr)], s=marker_size, c=else_c, alpha=alpha_val, edgecolors='none')
    # plot the scatter for quasr
    ax[ax_list[i][0], ax_list[i][1]].scatter(AEs[mask & mask_quasr], Qs[mask & mask_quasr], s=marker_size, c=quasr_c, alpha=alpha_val, edgecolors='none')

# add all data to the center plot
# plot the scatter for else
ax[1,1].scatter(AEs[(~mask_quasr)], Qs[(~mask_quasr)], s=marker_size, c=else_c, alpha=alpha_val, edgecolors='none')
# plot the scatter for quasr
ax[1,1].scatter(AEs[mask_quasr], Qs[mask_quasr], s=marker_size, c=quasr_c, alpha=alpha_val, edgecolors='none')

# Add two empty points for the legend
ax[1, 1].scatter([], [], s=1, c=quasr_c, alpha=1, label=r'\textsc{quasr}')
ax[1, 1].scatter([], [], s=1, c=else_c, alpha=1, label=r'other')
ax[1, 1].legend(loc='lower left', fontsize=6)

# add text to the top left
for i, mask in enumerate(mask_list):
    ax[ax_list[i][0], ax_list[i][1]].text(0.05, 0.9, nfp_labels[i], transform=ax[ax_list[i][0], ax_list[i][1]].transAxes, fontsize=12)

# Add lnA = 2/3 * lnQ power law to all plots
log10Q = np.linspace(-2, 3, 100)
log10A = 2/3 * log10Q - 2.0

# Plot the power law on each subplot
for i in range(3):
    for j in range(3):
        ax[i, j].plot(log10A, log10Q, 'k--', lw=1)

# add 'all' to the center plot
ax[1,1].text(0.05, 0.9, r'All', transform=ax[1,1].transAxes, fontsize=12)

# add labels
ax[2,1].set_xlabel(r'$\log_{10} \widehat{A}$')
ax[1,0].set_ylabel(r'$\log_{10} Q$')

# set lower ylim to -1
ax[1,1].set_ylim(-2, 3)
ax[1,1].set_xlim(-4,0)

# save the figure
plt.savefig('plots/scatter_AE_Q_random.png', dpi=1000)

plt.show()
plt.close('all')




######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################

angles = np.arctan2(w_Ts, w_ns)
dpdx = w_Ts + w_ns 

# Create a 2x2 grid
fac = 3/4
fig, ax = plt.subplots(2, 2, figsize=(fac*8, fac*8), constrained_layout=True)

# Top-left plot: Scatter w_N vs w_T, colored by Q
#Qs_cols = np.maximum(Qs, -2)  # Cap minimum value for better color scaling
# cap both min and max values for better color scaling
Qs_cols = np.clip(Qs, -2, 3)  # Cap minimum and maximum values for better color scaling

sc1 = ax[0, 0].scatter(w_ns, w_Ts, c=Qs_cols, s=0.2, edgecolors='none', cmap='viridis')
ax[0, 0].set_xlabel(r'$\hat{\omega}_n$')
ax[0, 0].set_ylabel(r'$\hat{\omega}_T$')

# Top-right plot: Scatter w_N vs w_T, colored by AE
AE_cols = np.maximum(AEs, -4)  # Cap minimum value for better color scaling
sc2 = ax[0, 1].scatter(w_ns, w_Ts, c=AE_cols, s=0.2, edgecolors='none', cmap='plasma')
ax[0, 1].set_xlabel(r'$\hat{\omega}_n$')
ax[0, 1].set_ylabel(r'$\hat{\omega}_T$')

# Bottom-left plot: Scatter angles vs Qs
sc3 = ax[1, 0].scatter(angles / np.pi, Qs, c=Qs_cols, s=0.2, edgecolors='none', cmap='viridis')
ax[1, 0].set_xlabel(r'$\arctan(\eta)/\pi$')
ax[1, 0].set_ylabel(r'$\log_{10} Q$')
cbar3 = fig.colorbar(sc3, ax=ax[1, 0], extend='both', orientation='horizontal', pad=0.1)
cbar3.set_label(r'$\log_{10} Q$')

# Bottom-right plot: Scatter angles vs AEs
sc4 = ax[1, 1].scatter(angles / np.pi, AEs, c=AE_cols, s=0.2, edgecolors='none', cmap='plasma')
ax[1, 1].set_xlabel(r'$\arctan(\eta)/\pi$')
ax[1, 1].set_ylabel(r'$\log_{10} \widehat{A}$')
cbar4 = fig.colorbar(sc4, ax=ax[1, 1], extend='min', orientation='horizontal', pad=0.1)
cbar4.set_label(r'$\log_{10} \widehat{A} $')

# Add a line where angle = arctan(2/3) only for the bottom row
for j in range(2):
    ax[1, j].axvline(x=np.arctan(2/3) / np.pi, color='r', linestyle='--', label=r'$\arctan(2/3)/\pi$', lw=1)



# Set range for y-axis in the top row
ax[0, 0].set_ylim(bottom=0)
ax[0, 1].set_ylim(bottom=0)
# set range for x-axis in top row
ax[0, 0].set_xlim(left=np.min(w_ns))
ax[0, 1].set_xlim(left=np.min(w_ns))
# add w_T=2/3*w_N line
for j in range(2):
    ax[0, j].plot(np.linspace(0, 100, 100), 2/3 * np.linspace(0, 100, 100), 'r--', lw=1)

# Set range for y-axis in the bottom row
ax[1, 0].set_ylim(-8,5)
# labels at -8,-4,0
ax[1, 0].set_yticks([-8,-4,0])
ax[1, 1].set_ylim(bottom=-6)

# # left label is outside the plot, so we need to adjust the position
# ax[1, 0].yaxis.set_label_coords(-0.2, 0.5)

# Save the figure
plt.savefig('plots/scatter_angles.png', dpi=1000)


plt.show()

