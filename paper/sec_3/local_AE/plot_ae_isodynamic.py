# Here we plot the AE. The plot is saved in the folder plots as AE.png
import source.AE_ITG as ae
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

# enable latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def n_significant_digits_ceil(x,n=1):
    # find the order of magnitude
    order = np.floor(np.log10(np.abs(x)))
    # divide out the order of magnitude
    x = x/10**order
    # round to n significant digits
    x = np.ceil(x*10**n)/10**n
    return x*10**order

def n_significant_digits_floor(x,n=1):
    # find the order of magnitude
    order = np.floor(np.log10(np.abs(x)))
    # divide out the order of magnitude
    x = x/10**order
    # round to n significant digits
    x = np.floor(x*10**n)/10**n
    return x*10**order

res=100

# span a range of values for w_n,w_T
w_n = np.linspace(-10.0,10.0,res)
w_T = np.linspace(-10.0,10.0,res)

# make a list of w_alpha and w_psi values
w_alpha = -1 * np.asarray([0.0,1.0,1.0])
w_psi   =      np.asarray([1.0,0.0,1.0])

# make a meshgrid for w_n w_T, and w_alpha
_, _, w_alpha   = np.meshgrid(w_n, w_T, w_alpha)
w_n, w_T, w_psi     = np.meshgrid(w_n, w_T, w_psi)

# make container for AE, k_alpha, k_psi
AE = np.zeros_like(w_alpha)
k_alpha = np.zeros_like(w_alpha)
k_psi = np.zeros_like(w_alpha)

# calculate the AE parallel
def calculate_AE_strong(w_alpha, w_psi, w_n, w_T,idx):
    print(idx)
    dict = ae.calculate_AE_iso(w_alpha[idx], w_n[idx], w_T[idx])
    return dict

# calculate the AE using multiprocessing
if __name__ == '__main__':
    mp.freeze_support()
    # get number of processors
    n_proc = mp.cpu_count()

    # calculate the AE in the isodynamic limit using multiprocessing
    with mp.Pool(n_proc) as pool:
        results = [pool.apply_async(calculate_AE_strong, args=(w_alpha,w_psi,w_n,w_T,idx)) for idx in np.ndindex(w_alpha.shape)]
        for idx, res in zip(np.ndindex(w_alpha.shape),results):
            dict = res.get()
            AE[idx] = dict['AE']
            k_alpha[idx] = dict['k_alpha']
            k_psi[idx] = 0.0

    # save results
    np.save('plots/data/AE_iso.npy',AE)
    np.save('plots/data/k_alpha_iso.npy',k_alpha)
    np.save('plots/data/k_psi_iso.npy',k_psi)

    # plot the AE, k_alpha, k_psi
    scaling_fac=3/4
    fig, ax = plt.subplots(3,3,figsize=(scaling_fac*8.0,scaling_fac*8.0),constrained_layout=True,sharex=True,sharey=True)
    # row 1 has the AE, row 2 the k_alpha, row 3 the k_psi
    # column corresponds to w_alpha and w_psi idx

    # find maximal and minimal AE, k_alpha, k_psi
    lvls_res    = 25
    AE_max      = n_significant_digits_ceil(np.nanmax(AE))
    AE_min      = 0.0
    k_alpha_max = n_significant_digits_ceil(np.nanmax(k_alpha))
    k_alpha_min = 0.0
    k_psi_max   = 1  # set these manually, since it's zero everywhere
    k_psi_min   = -1 # see above
    AE_lvls     = np.linspace(AE_min,AE_max,lvls_res)
    k_alpha_lvls= np.linspace(k_alpha_min,k_alpha_max,lvls_res)
    # if k_psi_max and min are nan, set them to 0
    k_psi_lvls  = np.sort(np.linspace(k_psi_min,k_psi_max,lvls_res))

    for i in range(3):
        ax[0,i].contourf(w_n[:,:,i],w_T[:,:,i],AE[:,:,i],levels=AE_lvls,cmap='gist_heat_r')
        ax[1,i].contourf(w_n[:,:,i],w_T[:,:,i],k_alpha[:,:,i],levels=k_alpha_lvls,cmap='viridis')
        ax[2,i].contourf(w_n[:,:,i],w_T[:,:,i],k_psi[:,:,i],levels=k_psi_lvls,cmap='viridis')


    # set labels, only on the left and bottom plots
    ax[2,0].set_xlabel(r'$\hat{\omega}_n$')
    ax[2,1].set_xlabel(r'$\hat{\omega}_n$')
    ax[2,2].set_xlabel(r'$\hat{\omega}_n$')
    ax[0,0].set_ylabel(r'$\hat{\omega}_T$')
    ax[1,0].set_ylabel(r'$\hat{\omega}_T$')
    ax[2,0].set_ylabel(r'$\hat{\omega}_T$')

    # add colorbars to rows, to the right of the rows
    cbar1 = fig.colorbar(ax[0,2].contourf(w_n[:,:,2],w_T[:,:,2],AE[:,:,2],levels=AE_lvls,cmap='gist_heat_r'),ax=ax[0,2])
    cbar2 = fig.colorbar(ax[1,2].contourf(w_n[:,:,2],w_T[:,:,2],k_alpha[:,:,2],levels=k_alpha_lvls,cmap='viridis'),ax=ax[1,2])
    cbar3 = fig.colorbar(ax[2,2].contourf(w_n[:,:,2],w_T[:,:,2],k_psi[:,:,2],levels=k_psi_lvls,cmap='viridis'),ax=ax[2,2])
    # add labels to the colorbars
    cbar1.set_label(r'$\widehat{A}$')
    cbar2.set_label(r'$\hat{\kappa}_{\alpha}$')
    cbar3.set_label(r'$\hat{\kappa}_{\psi}$')
    # set ticks only at max and min
    cbar1.set_ticks([AE_min,AE_max])
    cbar2.set_ticks([k_alpha_min,k_alpha_max])
    cbar3.set_ticks([k_psi_min,k_psi_max])

    

    plt.savefig('plots/AE_iso.png',dpi=1000)
    plt.show()