# here we plot the AE as a function of w_alpha and w_psi

import source.ae as sua
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# make grid of omega_alpha and omega_psi [-10,10]x[-10,10]
res = 100
w_range = 3.0
w_alpha = np.linspace(-2*w_range, +2*w_range, res)
w_psi   = np.linspace(-w_range, +w_range, res)
w_alpha_t, w_psi_t = np.meshgrid(w_alpha, w_psi)

# give values to omega_n and omega_T
w_n = 1.0
w_T = 3.0

# calculate AE for each point in the grid
AE_full = np.zeros_like(w_alpha_t)
kpsi_full = np.zeros_like(w_alpha_t)
kalpha_full = np.zeros_like(w_alpha_t)

def calculate_AE(idx):
    dict =  sua.calculate_AE_arr(w_T, w_n, np.asarray([w_alpha_t[idx]]), np.asarray([w_psi_t[idx]]))
    print(f'w_alpha = {w_alpha_t[idx]:+.2f}, w_psi = {w_psi_t[idx]:+.2f}, AE = {dict["AE"][0]:+.2f}', end='\r')
    return dict

# do the calculation in parallel
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(calculate_AE, [idx for idx, _ in np.ndenumerate(w_alpha_t)])
    pool.close()
    pool.join()

    # unpack results and store in array
    for idx, vals in enumerate(results):
        i, j = np.unravel_index(idx, w_alpha_t.shape)
        AE_full[i, j] = vals["AE"][0]
        kpsi_full[i, j] = vals["k_psi"][0]
        kalpha_full[i, j] = vals["k_alpha"][0]

    # plot the results
    import matplotlib.pyplot as plt
    scale = 3/4
    lvl_res = 30
    AE_full = np.log10(AE_full)
    AE_max = np.ceil(AE_full.max())
    AE_min = -3.0
    print('\n')
    print(f"AE_min = {AE_full.min()}")
    print(f"AE_max = {AE_full.max()}")
    kpsi_max = np.abs(kpsi_full).max()
    kalpha_max = np.abs(kalpha_full).max()
    # round up to nearest 0.1
    kpsi_max = np.ceil(kpsi_max*10)/10
    kalpha_max = np.ceil(kalpha_max*10)/10
    lvls_AE = np.linspace(AE_min, AE_max, lvl_res)
    lvls_kpsi = np.linspace(-kpsi_max, kpsi_max, lvl_res)
    lvls_kalpha = np.linspace(-kalpha_max, kalpha_max, lvl_res)

    fig, ax = plt.subplots(1, 3, figsize=(scale*8.0, scale*8.0/2.4), constrained_layout=True)
    
    # Plot AE_full
    cf0 = ax[0].contourf(w_psi_t, w_alpha_t, AE_full, levels=lvls_AE, cmap='gist_heat_r', extend='min')
    cbar0 = plt.colorbar(cf0, ax=ax[0], orientation='horizontal', location='top', ticks=[AE_min, AE_max])
    cbar0.set_label(r'$\log_{10} \widehat{A}$')
    ax[0].set_xlabel(r'$\hat{\omega}_{\psi}$')
    ax[0].set_ylabel(r'$\hat{\omega}_{\alpha}$')
    
    # Plot kpsi_full
    cf1 = ax[2].contourf(w_psi_t, w_alpha_t, kpsi_full, levels=lvls_kpsi, cmap='PiYG')
    cbar1 = plt.colorbar(cf1, ax=ax[2], orientation='horizontal', location='top', ticks=[-kpsi_max, 0, kpsi_max])
    cbar1.set_label(r'$\hat{\kappa}_{\psi}$')
    ax[2].set_xlabel(r'$\hat{\omega}_{\psi}$')
    ax[2].set_yticklabels([])  # Suppress y labels
    ax[2].set_ylabel(None) # Suppress y labels
    
    # Plot kalpha_full
    cf2 = ax[1].contourf(w_psi_t, w_alpha_t, kalpha_full, levels=lvls_kalpha, cmap='RdBu_r')
    cbar2 = plt.colorbar(cf2, ax=ax[1], orientation='horizontal', location='top', ticks=[-kalpha_max, 0, kalpha_max])
    cbar2.set_label(r'$\hat{\kappa}_{\alpha}$')
    ax[1].set_xlabel(r'$\hat{\omega}_{\psi}$')
    ax[1].set_yticklabels([])  # Suppress y labels
    ax[1].set_ylabel(None) # Suppress y labels

    # add a plot of the path of a circular tokamak (walpha = cos(theta), wpsi = sin(theta))
    # theta = np.linspace(-np.pi, np.pi, 100)
    # a = 2.0
    # w_psi_path = a*np.sin(theta)
    # w_alpha_path = a*np.cos(theta) 
    # ax[0].plot(w_psi_path, w_alpha_path, 'w')

    # save the figure
    plt.savefig('plots/AE_geometry_dependence.png', dpi=1000)

    plt.show()
