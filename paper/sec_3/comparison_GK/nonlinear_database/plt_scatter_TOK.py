# Here the scatter plot with L_con is created

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
# keep files of form TOK_*_FIXED.hdf5
files = [f for f in files if f.startswith('TOK_') and f.endswith('FIXED.hdf5')]
# sort the files
files.sort()
# initialize the data
AEs = []
Qs = []
nfps = []
w_ns = []
w_Ts = []
L_cons=[]
R = []

# loop over the files
for f in files:
    # load the data
    with h5py.File(path+'/'+f, 'r') as hf:
        # get the data
        data = hf
        # loop over the tubes with a progress bar
        # print length of data
        print(f'Number of tubes in {f}: {len(data.keys())}')
        for tube in tqdm.tqdm(data.keys(), desc=f"Processing {f}"):
            # get the data
            AE = data[tube]['AE_val'][()]
            Q = data[tube]['Q'][()]
            nfp = data[tube]['nfp'][()]
            w_n = data[tube]['w_n'][()]
            w_T = data[tube]['w_T'][()]
            L_con = data[tube]['L_con'][()]
            Rmaj = data[tube]['Rmaj'][()]
            # append to the list
            AEs.append(AE)
            Qs.append(Q)
            nfps.append(nfp)
            w_ns.append(w_n)
            w_Ts.append(w_T)
            L_cons.append(L_con)
            R.append(Rmaj)




# convert to numpy arrays
AEs = np.log10(np.array(AEs))
Qs = np.log10(np.array(Qs))
L_cons = np.array(L_cons)  # Keep L_cons linear
# only keep Rmaj between 9.0 and 10.0
R = np.array(R)

Rmin = 1.0
Rmax = 10.0
Lmin = 1.0
Lmax = 100.0
AEs = AEs[(R > Rmin) & (R < Rmax)]
Qs = Qs[(R > Rmin) & (R < Rmax)]
L_cons = L_cons[(R > Rmin) & (R < Rmax)]
R = R[(R > Rmin) & (R < Rmax)]
# filter out L_cons < threshold
threshold = 1.0
AEs = AEs[L_cons > threshold]
Qs = Qs[L_cons > threshold]
L_cons = L_cons[L_cons > threshold]

# check if L_cons should be plotted logarithmically
log_L = True  # Set this to False if you don't want logarithmic scaling
if log_L:
    L_cons = np.log10(L_cons)
    Lmin = np.log10(Lmin)
    Lmax = np.log10(Lmax)
    print(Lmin, Lmax)

# add 3/2 power law
lnQ = np.linspace(-2, 3, 100)
lnA = 2/3 * lnQ - 2.0

# scatter
fac = 3/4/2
fig, ax = plt.subplots(figsize=(fac*8, fac*8*0.75), constrained_layout=True)
sc = ax.scatter(AEs, Qs, s=0.8, c=L_cons, cmap='CMRmap_r', alpha=1.0, edgecolors='none', vmin=Lmin, vmax=Lmax)
# set ticklabels
if log_L:
    cbar = fig.colorbar(sc, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels([r'$0$', r'$1$', r'$2$'])
else:
    cbar = fig.colorbar(sc, ax=ax, ticks=[1, 50, 100])
    cbar.set_ticklabels([r'$1$', r'$50$', r'$100$'])
cbar.solids.set_alpha(1)
cbar.set_label(r'$\log_{10}(L_{\mathrm{con}}/a_{\rm minor})$' if log_L else r'$L_{\mathrm{con}}/a_{\rm minor}$')
ax.plot(lnA, lnQ, 'k--', label=r'$Q \propto A^{3/2}$')
ax.set_xlabel(r'$\log_{10} \widehat{A} $')
ax.set_ylabel(r'$\log_{10} Q$')
ax.set_ylim(-2, 3)
ax.set_xlim(-4, 0)

# save the figure
plt.savefig('plots/scatter_AE_Q_L.png', dpi=1000)

plt.show()
plt.close()

# Create histograms for AE
fig, ax = plt.subplots(figsize=(fac*8, fac*8), constrained_layout=True)

# Filter AE values based on Q
Q_threshold = -1 
AE_Q_less = AEs[Qs < Q_threshold]
AE_Q_greater = AEs[Qs >= Q_threshold]

# Plot histograms
bins = np.linspace(-4, 0, 50)  # Define bins for the histogram
ax.hist(AE_Q_less, bins=bins, alpha=0.7, label=r'$Q < 0.1$', color='blue', density=True)
ax.hist(AE_Q_greater, bins=bins, alpha=0.7, label=r'$Q \geq 0.1$', color='orange', density=True)

# Add labels and legend
ax.set_xlabel(r'$\log_{10} A$')
ax.set_ylabel('Frequency')
ax.legend()
ax.set_title(r'Histograms of $\log_{10} A$ for Different $Q$ Ranges')

plt.show()
plt.close()