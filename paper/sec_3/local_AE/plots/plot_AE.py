    # # save results
    # np.save('plots/data/AE.npy',AE)
    # np.save('plots/data/k_alpha.npy',k_alpha)
    # np.save('plots/data/k_psi.npy',k_psi)

    # # plot the AE, k_alpha, k_psi
    # scaling_fac=3/4
    # fig, ax = plt.subplots(3,3,figsize=(scaling_fac*8.0,scaling_fac*8.0),constrained_layout=True,sharex=True,sharey=True)
    # # row 1 has the AE, row 2 the k_alpha, row 3 the k_psi
    # # column corresponds to w_alpha and w_psi idx

    # # find maximal and minimal AE, k_alpha, k_psi
    # lvls_res    = 25
    # AE_max      = n_significant_digits_ceil(np.nanmax(AE))
    # AE_min      = 0.0
    # k_alpha_max = n_significant_digits_ceil(np.nanmax(k_alpha))
    # k_alpha_min = n_significant_digits_floor(np.nanmin(k_alpha))
    # k_psi_max   = n_significant_digits_ceil(np.nanmax(k_psi))
    # k_psi_min   = n_significant_digits_floor(np.nanmin(k_psi))
    # AE_lvls     = np.linspace(AE_min,AE_max,lvls_res)
    # k_alpha_lvls= np.linspace(k_alpha_min,k_alpha_max,lvls_res)
    # k_psi_lvls  = np.linspace(k_psi_min,k_psi_max,lvls_res)

    # for i in range(3):
    #     ax[0,i].contourf(w_n[:,:,i],w_T[:,:,i],AE[:,:,i],levels=AE_lvls,cmap='gist_heat_r')
    #     ax[1,i].contourf(w_n[:,:,i],w_T[:,:,i],k_alpha[:,:,i],levels=k_alpha_lvls,cmap='viridis')
    #     ax[2,i].contourf(w_n[:,:,i],w_T[:,:,i],k_psi[:,:,i],levels=k_psi_lvls,cmap='viridis')


    # # set labels, only on the left and bottom plots
    # ax[2,0].set_xlabel(r'$\hat{\omega}_n$')
    # ax[2,1].set_xlabel(r'$\hat{\omega}_n$')
    # ax[2,2].set_xlabel(r'$\hat{\omega}_n$')
    # ax[0,0].set_ylabel(r'$\hat{\omega}_T$')
    # ax[1,0].set_ylabel(r'$\hat{\omega}_T$')
    # ax[2,0].set_ylabel(r'$\hat{\omega}_T$')

    # # add colorbars to rows, to the right of the rows
    # cbar1 = fig.colorbar(ax[0,2].contourf(w_n[:,:,2],w_T[:,:,2],AE[:,:,2],levels=AE_lvls,cmap='gist_heat_r'),ax=ax[0,2])
    # cbar2 = fig.colorbar(ax[1,2].contourf(w_n[:,:,2],w_T[:,:,2],k_alpha[:,:,2],levels=k_alpha_lvls,cmap='viridis'),ax=ax[1,2])
    # cbar3 = fig.colorbar(ax[2,2].contourf(w_n[:,:,2],w_T[:,:,2],k_psi[:,:,2],levels=k_psi_lvls,cmap='viridis'),ax=ax[2,2])
    # # add labels to the colorbars
    # cbar1.set_label(r'$\widehat{A}$')
    # cbar2.set_label(r'$\hat{\kappa}_{\alpha}$')
    # cbar3.set_label(r'$\hat{\kappa}_{\psi}$')
    # # set ticks only at max and min
    # cbar1.set_ticks([AE_min,AE_max])
    # cbar2.set_ticks([k_alpha_min,k_alpha_max])
    # cbar3.set_ticks([k_psi_min,k_psi_max])

    

    # plt.savefig('plots/AE.png',dpi=1000)
    # plt.show()


# here we adjust the above script to read AE.npy, k_alpha.npy, k_psi.npy and plot them
import numpy as np
import matplotlib.pyplot as plt

# enable LaTeX
plt.rc('text',usetex=True)
plt.rc('font',family='serif')

plot = True

if plot:
    # load the data
    AE = np.load('data/AE.npy')
    k_alpha = np.load('data/k_alpha.npy')
    k_psi = np.load('data/k_psi.npy')

    # w_T and w_n go from -10 to 10
    w_n = np.linspace(-10.0,10.0,100)
    w_T = np.linspace(-10.0,10.0,100)

    w_n, w_T = np.meshgrid(w_n,w_T)

    # make a figure
    scale_fac=3/4
    fig, ax = plt.subplots(3,3,figsize=(scale_fac*8.0,scale_fac*7.0),constrained_layout=True)
    # row 1 has the AE, row 2 the k_alpha, row 3 the k_psi
    # column corresponds to w_alpha and w_psi idx

    # find maximal and minimal AE, k_alpha, k_psi
    lvls_res    = 25
    AE_max      = np.nanmax(AE)
    AE_min      = 0.0
    k_alpha_max = np.nanmax(k_alpha)
    k_alpha_min = np.nanmin(k_alpha)
    k_psi_max   = np.nanmax(k_psi)
    k_psi_min   = np.nanmin(k_psi)

    # override AE_min and AE_max to 0,1.9
    # and k_alpha_min k_alpha_max -10,10
    # and k_psi_min k_psi_max -1.8,1.8
    AE_min = 0
    AE_max = 1.9
    k_alpha_min = -15
    k_alpha_max = 15
    k_psi_min = -1.8
    k_psi_max = 1.8

    AE_lvls     = np.linspace(AE_min,AE_max,lvls_res)
    k_alpha_lvls= np.linspace(k_alpha_min,k_alpha_max,lvls_res)
    k_psi_lvls  = np.linspace(k_psi_min,k_psi_max,lvls_res)

    for i in range(3):
        ax[0,i].contourf(w_n,w_T,AE[:,:,i],levels=AE_lvls,cmap='gist_heat_r')
        ax[1,i].contourf(w_n,w_T,k_alpha[:,:,i],levels=k_alpha_lvls,cmap='bwr',extend='both')
        ax[2,i].contourf(w_n,w_T,k_psi[:,:,i],levels=k_psi_lvls,cmap='bwr')


    # set labels, only on the left and bottom plots
    ax[2,0].set_xlabel(r'$\hat{\omega}_n$')
    ax[2,1].set_xlabel(r'$\hat{\omega}_n$')
    ax[2,2].set_xlabel(r'$\hat{\omega}_n$')
    ax[0,0].set_ylabel(r'$\hat{\omega}_T$')
    ax[1,0].set_ylabel(r'$\hat{\omega}_T$')
    ax[2,0].set_ylabel(r'$\hat{\omega}_T$')

    # add colorbars to rows, to the right of the rows
    cbar1 = fig.colorbar(ax[0,2].contourf(w_n,w_T,AE[:,:,2],levels=AE_lvls,cmap='gist_heat_r'),ax=ax[0,2])
    # extend the colorbar to include arrows, only at the max for k_alpha
    cbar2 = fig.colorbar(ax[1,2].contourf(w_n,w_T,k_alpha[:,:,2],levels=k_alpha_lvls,cmap='bwr',extend='both'),ax=ax[1,2])
    cbar3 = fig.colorbar(ax[2,2].contourf(w_n,w_T,k_psi[:,:,2],levels=k_psi_lvls,cmap='bwr'),ax=ax[2,2])
    # add labels to the colorbars
    cbar1.set_label(r'$\widehat{A}$')
    cbar2.set_label(r'$\hat{\kappa}_{\alpha}$')
    cbar3.set_label(r'$\hat{\kappa}_{\psi}$')
    # set ticks only at max and min
    cbar1.set_ticks([AE_min,AE_max])
    cbar2.set_ticks([k_alpha_min,k_alpha_max])
    cbar3.set_ticks([k_psi_min,k_psi_max])

    # add titles to the columns
    ax[0,0].set_title(r'$(\hat{\omega}_{\alpha},\hat{\omega}_{\psi}) = (0,1)$')
    ax[0,1].set_title(r'$(\hat{\omega}_{\alpha},\hat{\omega}_{\psi}) = (-1,0)$')
    ax[0,2].set_title(r'$(\hat{\omega}_{\alpha},\hat{\omega}_{\psi}) = (-1,1)$')


    # add text (a), (b), (c), ... to the plots, from top left to bottom right
    text_arr = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    for i in range(3):
        for j in range(3):
            # be sure to center the text at given position
            ax[i,j].text(-7.,-8.,text_arr[3*i+j],fontsize=12,color='black',va='center',ha='center')

    plt.savefig('AE.png',dpi=1000)
    plt.show()