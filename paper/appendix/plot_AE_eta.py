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


def plot_routine(w_alpha, w_n_inv, eta, AE_iso):
   # do log scale
    AE_iso_plot = np.log10(AE_iso)
    # replace nan with -inf
    AE_iso_plot = np.nan_to_num(AE_iso_plot,neginf=-np.inf)
    # find the maximum value
    print('max value of AE_iso_plot:',np.max(AE_iso_plot))
    max_val = np.max(AE_iso_plot)
    print('max value in plot set to:',max_val)
    min_val = max_val-5

    # extract w_alpha_str from w_alpha
    w_alpha_str = [rf'{val}' for val in w_alpha[:,0,0]]

    # plot the AE in the isodynamic limit
    fig, ax = plt.subplots(2, 2, figsize=(6.0,5.0), constrained_layout=True,sharex=True,sharey=True)


    for i in range(4):
        levels = np.linspace(min_val,max_val,21)
        ax[i//2,i%2].contourf(w_n_inv[i,:,:],eta[i,:,:],AE_iso_plot[i,:,:],cmap='gist_heat_r',levels=levels,extend='min')
        ax[i//2,i%2].set_title(r'$\hat{\omega}_{\alpha}= '+w_alpha_str[i]+'$')
        ax[i//2,i%2].set_xlabel(r'$1/\hat{\omega}_{n}$')
        ax[i//2,i%2].set_ylabel(r'$\eta$')

        # set plot limits
        ax[i//2,i%2].set_xlim(w_n_inv[i,0,0],w_n_inv[i,-1,0])
        ax[i//2,i%2].set_ylim(eta[i,0,0],eta[i,0,-1])

        # add dashed line at eta = 2/3
        ax[i//2,i%2].plot([w_n_inv[i,0,0],w_n_inv[i,-1,0]],[2/3,2/3],'k--')



    # add colorbar, ticks at max_val, min_val
    cbar = fig.colorbar(ax[0,0].collections[0], ax=ax, orientation='vertical',ticks=[min_val,max_val])
    cbar.set_label(r'$\log_{10} \widehat{A}$')

    # save the plot
    plt.savefig('plots/AE_eta_iso.png', dpi=ps.dpi)

    plt.show()


if recalculate:
    # make grid in w_n, eta
    n_res = 100
    w_n_inv = np.linspace(0.01,1.5,n_res)
    eta = np.linspace(0,4.5,n_res)

    # choose four values for w_alpha 
    w_alpha = -1.0*np.array([1, 2, 3, 6])


    # meshgrid
    w_alpha, w_n_inv, eta = np.meshgrid(w_alpha, w_n_inv, eta, indexing='ij')

    # initialize AE_iso
    AE_iso = np.zeros(w_n_inv.shape)

    # Function to calculate AE for a given index
    def calculate_ae(idx):
        return ae.available_energy_iso(w_alpha[idx], 1/w_n_inv[idx], eta[idx]/w_n_inv[idx])



    if __name__ == '__main__':
        mp.freeze_support()

        n_proc = 8

        print(f"Using {n_proc} processes")

        # calculate the AE in the isodynamic limit using multiprocessing
        with mp.Pool(n_proc) as pool:
            results = pool.map(calculate_ae, np.ndindex(w_n_inv.shape))

        # Fill AE_iso with the results
        for idx, result in zip(np.ndindex(w_n_inv.shape), results):
            AE_iso[idx] = result

        # save solution

        
        # plot the results
        plot_routine(w_alpha, w_n_inv, eta, AE_iso)