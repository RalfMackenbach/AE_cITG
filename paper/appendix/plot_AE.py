# Here we plot the AE in the isodynamic limit. The plot is saved in the folder plots as AE_iso.png
import source.AE_ITG as ae
import numpy as np
import multiprocessing as mp
import source.plot_settings as ps
import matplotlib.pyplot as plt

# enable latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


recalculate = True


def plot_routine(w_alpha, w_n, w_T, AE_iso, k_alpha):
    # set 0.0 to np.nan in AE_iso
    AE_iso[AE_iso == 0.0] = np.nan 
    # do log scale
    AE_iso_plot = np.log10(AE_iso)
    k_alpha = np.log10(k_alpha)
    #set nan in k_alpha to -inf
    k_alpha = np.nan_to_num(k_alpha,neginf=-np.inf)
    # make mask for NaN values
    mask = np.isnan(AE_iso_plot)
    # where there are NaN values set k_alpha to NaN
    k_alpha[mask] = np.nan
    # find the maximum value
    print(np.nanmax(AE_iso_plot))  
    max_val_AE = np.ceil(np.nanmax(AE_iso_plot))
    min_val_AE = max_val_AE-3
    # also find maximum value for k_alpha
    max_val_k_alpha = np.ceil(np.nanmax(k_alpha))
    min_val_k_alpha = max_val_k_alpha-6
    print()

    # make array with (a), (b), (c), (d)
    ax_label = [r'(a)',r'(b)',r'(c)',r'(d)']

    # plot the AE in the isodynamic limit
    scaling_fac = 3/4
    fig, ax = plt.subplots(4, 2, figsize=(scaling_fac*6.0,scaling_fac*10.0), constrained_layout=True,sharex=True,sharey=True)

    index_arr = np.zeros_like(ax)

    for idx, val in np.ndenumerate(index_arr):
        
        # if [0,0], [0,1], [1,0] or [1,1] plot AE_iso
        # if [2,0], [2,1], [3,0] or [3,1] plot k_alpha
        if idx[0] < 2:
            # first index is binary count of ax index
            first_index = idx[1] * 1 + idx[0] * 2
            levels = np.linspace(min_val_AE,max_val_AE,21)
            ax[idx].contourf(w_n[first_index,:,:],w_T[first_index,:,:],AE_iso_plot[first_index,:,:],cmap='gist_heat_r',levels=levels,extend='min')
            # set text at the top-left corner of the plot
            ax[idx].text(0.05,0.9,ax_label[first_index],transform=ax[idx].transAxes, color='white')
        else:
            # first index is binary count of ax index, offset by 2
            first_index = idx[1] * 1 + (idx[0]-2) * 2
            levels = np.linspace(min_val_k_alpha,max_val_k_alpha,21)
            ax[idx].contourf(w_n[first_index,:,:],w_T[first_index,:,:],k_alpha[first_index,:,:],cmap='viridis',levels=levels,extend='min')
            # set text at the top-left corner of the plot
            ax[idx].text(0.05,0.9,ax_label[first_index],transform=ax[idx].transAxes, color='white')
        # arch our stability region, (w_T = -omega_alpha), (w_T = 0), (w_T = 2/3).
        # intersection between w_T = -w_alpha and w_T = 2/3*w_n is at w_n = -3/2*w_alpha
        w_n_max = np.max(w_n[first_index,:,:])
        w_intersec = -3/2*w_alpha[first_index,0,0]
        # plot line horizontal line at -w_alpha from w_n = w_intersec to w_n = w_n_max
        ax[idx].plot([w_intersec,w_n_max],[-w_alpha[first_index,0,0],-w_alpha[first_index,0,0]],'k')
        # plot w_T = 2/3*w_n line from w_intersec to w_n = w_n_max
        ax[idx].plot([w_intersec,w_n_max],[2/3*w_intersec,2/3*w_n_max],'k')
        # plot w_T = 0 line from w_n = w_n_min to w_n=0
        w_n_min = np.min(w_n[first_index,:,:])
        ax[idx].plot([w_n_min,0],[0,0],'k')
        # plot w_T = 2/3 line from w_n_min to w_n = 0
        ax[idx].plot([w_n_min,0],[2/3*w_n_min,0],'k')
        # add line of pure density gradient from 0 to w_n_max
        ax[idx].plot([0,w_n_max],[0,0],'k--')



    # set x label for bottom plots, and y label for left plots
    for idx, _ in np.ndenumerate(index_arr):
        if idx[0] == 3:
            ax[idx].set_xlabel(r'$\hat{\omega}_{n}$')
        if idx[1] == 0:
            ax[idx].set_ylabel(r'$\hat{\omega}_{T}$')

    # add colorbar for AE_iso [0:2,:] and k_alpha [2:,:]
    cbar = fig.colorbar(ax[0,0].collections[0], ax=ax[0:2,:], orientation='vertical',ticks=[min_val_AE,max_val_AE])
    cbar.set_label(r'$\log_{10} \widehat{A}$')
    cbar = fig.colorbar(ax[2,0].collections[0], ax=ax[2:,:], orientation='vertical',ticks=[min_val_k_alpha,max_val_k_alpha])
    cbar.set_label(r'$\log_{10} \tilde{\kappa}_{\alpha}$')

    # save the plot
    plt.savefig('plots/AE_iso.png', dpi=ps.dpi)

    plt.show()


if recalculate:
    # make grid in w_n, w_T
    n_res = 100
    w_n = np.linspace(-10,10,n_res)
    w_T = np.linspace(-10,10,n_res)

    # choose four values for w_alpha 
    w_alpha = -1.0*np.array([1, 2, 3, 6])

    # meshgrid
    w_alpha, w_n, w_T = np.meshgrid(w_alpha, w_n, w_T, indexing='ij')

    # initialize AE_iso
    AE_iso = np.zeros(w_n.shape)

    # initialize k_alpha
    k_alpha = np.zeros(w_n.shape)

    # Function to calculate AE for a given index
    def calculate_ae(idx):
        return ae.available_energy_iso(w_alpha[idx], w_n[idx], w_T[idx])

    def calculate_k_alpha(idx):
        return ae.solve_k_alpha_iso(w_alpha[idx], w_n[idx], w_T[idx])


    if __name__ == '__main__':
        mp.freeze_support()

        n_proc = 8

        print(f"Using {n_proc} processes")

        # calculate the AE in the isodynamic limit using multiprocessing
        with mp.Pool(n_proc) as pool:
            results = pool.map(calculate_ae, np.ndindex(w_n.shape))
            results_k_alpha = pool.map(calculate_k_alpha, np.ndindex(w_n.shape))

        # Fill AE_iso with the results
        for idx, result in zip(np.ndindex(w_n.shape), results):
            if type(result) == np.ndarray:
                result = result[0]
            AE_iso[idx] = result
        
        # Fill k_alpha with the results
        for idx, result in zip(np.ndindex(w_n.shape), results_k_alpha):
            # check if result is array or list, and convert to scalar
            if type(result) == np.ndarray:
                result = result[0]
            k_alpha[idx] = result

        
        # plot the results
        plot_routine(w_alpha, w_n, w_T, AE_iso, k_alpha)