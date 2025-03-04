import source.ae as sua
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# resolution
res = 50

# parallel or not
par = True

# test solve k
w_alpha =np.asarray([0.0,-1.0,-1.0])
w_psi = np.asarray([1.0,0.0,1.0])

# make grid of w_n and w_T [-10,10]x[-10,10]
w_n = np.linspace(-10,10,res)
w_T = np.linspace(-10,10,res)
w_n_t, w_T_t, w_alpha_t = np.meshgrid(w_n, w_T, w_alpha)
w_n_t, w_T_t, w_psi_t = np.meshgrid(w_n, w_T, w_psi)

# kpsi and kalpha and AE arrs
AE_full, AE_strong, AE_iso = np.zeros_like(w_n_t), np.zeros_like(w_n_t), np.zeros_like(w_n_t)
k_psi_full, k_psi_strong, k_psi_iso = np.zeros_like(w_n_t), np.zeros_like(w_n_t), np.zeros_like(w_n_t)
k_alpha_full, k_alpha_strong, k_alpha_iso = np.zeros_like(w_n_t), np.zeros_like(w_n_t), np.zeros_like(w_n_t)

# hold solutions
k_psi = np.zeros((res,res))
k_alpha = np.zeros((res,res))
AE = np.zeros((res,res))

# solve for each point parallel
def get_dicts(idx):
    dict_full    = sua.calculate_AE_arr(w_T_t[idx], w_n_t[idx], w_alpha_t[idx], w_psi_t[idx])
    dict_strong  = sua.calculate_AE_strong_arr(w_T_t[idx], w_n_t[idx], w_alpha_t[idx], w_psi_t[idx])
    dict_iso     = sua.calculate_AE_iso_arr(w_T_t[idx], w_n_t[idx], w_alpha_t[idx], w_psi_t[idx])
    #print(f'w_n = {w_n_t[idx]:+.2f}, w_T = {w_T_t[idx]:+.2f}, AE_full = {dict_full["AE"]:+.2f}, AE_strong = {dict_strong["AE"]:+.2f}, AE_iso = {dict_iso["AE"]:+.2f}')
    print(f'w_n = {w_n_t[idx]:+.2f}, w_T = {w_T_t[idx]:+.2f}, w_alpha = {w_alpha_t[idx]:+.2f}, w_psi = {w_psi_t[idx]:+.2f}, AE_full = {dict_full["AE"]:+.2f}, AE_strong = {dict_strong["AE"]:+.2f}, AE_iso = {dict_iso["AE"]:+.2f}', end='\r')
    return [dict_full, dict_strong, dict_iso]

    
if par:
    if __name__ == '__main__':
        start_time = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(get_dicts, [idx for idx, _ in np.ndenumerate(w_n_t)])
        pool.close()
        pool.join()
        end_time = time.time()
        print('\n')
        print(f"Calculation completed in {end_time - start_time:.2f} seconds")

        # unpack results, and store in arrays
        for idx, vals in zip([idx for idx, _ in np.ndenumerate(w_n_t)], results):
            i, j, k = idx
            AE_full[i,j,k] = vals[0]["AE"]
            AE_strong[i,j,k] = vals[1]["AE"]
            AE_iso[i,j,k] = vals[2]["AE"]
            k_psi_full[i,j,k] = vals[0]["k_psi"]
            k_psi_strong[i,j,k] = vals[1]["k_psi"]
            k_psi_iso[i,j,k] = vals[2]["k_psi"]
            k_alpha_full[i,j,k] = vals[0]["k_alpha"]
            k_alpha_strong[i,j,k] = vals[1]["k_alpha"]
            k_alpha_iso[i,j,k] = vals[2]["k_alpha"]

        # plot the AE results on 3X3 grid
        lvl_res = 25
        lvls_AE = np.linspace(0.0, 2.2, lvl_res)
        cmap_AE = 'gist_heat_r'
        fac = 3/4
        fig, ax = plt.subplots(3,3, figsize=(fac*8,fac*8), constrained_layout=True, sharex=True, sharey=True)
        ax[0,0].contourf(w_n, w_T, AE_full[:,:,0], levels=lvls_AE, cmap=cmap_AE)
        ax[0,1].contourf(w_n, w_T, AE_full[:,:,1], levels=lvls_AE, cmap=cmap_AE)
        ax[0,2].contourf(w_n, w_T, AE_full[:,:,2], levels=lvls_AE, cmap=cmap_AE)
        ax[1,0].contourf(w_n, w_T, AE_strong[:,:,0], levels=lvls_AE, cmap=cmap_AE)
        ax[1,1].contourf(w_n, w_T, AE_strong[:,:,1], levels=lvls_AE, cmap=cmap_AE)
        ax[1,2].contourf(w_n, w_T, AE_strong[:,:,2], levels=lvls_AE, cmap=cmap_AE)
        ax[2,0].contourf(w_n, w_T, AE_iso[:,:,0], levels=lvls_AE, cmap=cmap_AE)
        ax[2,1].contourf(w_n, w_T, AE_iso[:,:,1], levels=lvls_AE, cmap=cmap_AE)
        ax[2,2].contourf(w_n, w_T, AE_iso[:,:,2], levels=lvls_AE, cmap=cmap_AE)

        # y ticks only on left plots [-10,10], x ticks only on bottom plots [-10,10]
        for i in range(3):
            ax[i,0].set_ylabel(r'$\hat{\omega}_T$')
            ax[i,0].set_yticks([-10,10])
            ax[i,0].set_yticklabels([-10,10])
            ax[2,i].set_xlabel(r'$\hat{\omega}_n$')
            ax[2,i].set_xticks([-10,10])
            ax[2,i].set_xticklabels([-10,10])

        # add colorbars to rows, to the right of the rows
        cbar_AE = fig.colorbar(ax[0,2].collections[0], ax=ax[0,2], ticks=[0.0, 2.2], label=r'$\widehat{A}$')
        cbar_AE.set_ticklabels([0.0, 2.2])
        cbar_AE_strong = fig.colorbar(ax[1,2].collections[0], ax=ax[1,2], ticks=[0.0, 2.2], label=r'$\widehat{A}_{\rm strong}$')
        cbar_AE_strong.set_ticklabels([0.0, 2.2])
        cbar_AE_iso = fig.colorbar(ax[2,2].collections[0], ax=ax[2,2], ticks=[0.0, 2.2], label=r'$\widehat{A}_{\rm iso}$')
        cbar_AE_iso.set_ticklabels([0.0, 2.2])

        # titles to columns indicating w_alpha and w_psi
        ax[0,0].set_title(r'$(\hat{\omega}_\alpha,\hat{\omega}_\psi) = (0,1)$')
        ax[0,1].set_title(r'$(\hat{\omega}_\alpha,\hat{\omega}_\psi) = (-1,0)$')
        ax[0,2].set_title(r'$(\hat{\omega}_\alpha,\hat{\omega}_\psi) = (-1,1)$')

        # add labels to the plots
        labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
        for i, ax in enumerate(ax.flat):
            # plot at bottom left of plot, in black
            ax.text(0.05, 0.05, labels[i
            ], transform=ax.transAxes, fontsize=12, va='bottom', ha='left', color='black')

        plt.savefig('plots/comparison_of_limits.png',dpi=1000)
