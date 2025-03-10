import numpy as np
import multiprocessing as mp
import source.utils_ae_weak as sua

# resolution
res = 30

# parallel or not
par = True

# test solve k
w_alpha = -1.0
w_psi =  1.0

# make grid of w_n and w_T [-10,10]x[-10,10]
w_n = np.linspace(-10,10,res)
w_T = np.linspace(-10,10,res)
w_n, w_T = np.meshgrid(w_n, w_T)

# hold solutions
k_psi = np.zeros((res,res))
k_alpha = np.zeros((res,res))
AE = np.zeros((res,res))

# solve for each point parallel
def solve_k(idx):
    i, j = idx
    k_psi_val, k_alpha_val = sua.solve_k_weak(w_alpha, w_psi, w_n[i,j], w_T[i,j], method='iterative', abs_tol=1e-5, rel_tol=1e-5)
    AE_val = sua.AE_local_weak(w_n[i,j], w_T[i,j], w_alpha, w_psi, k_alpha_val, k_psi_val)
    print(f'w_n = {w_n[i,j]:+.2f}, w_T = {w_T[i,j]:+.2f}, k_psi = {k_psi_val:+.2f}, k_alpha = {k_alpha_val:+.2f}, AE = {AE_val:+.2f}')
    return k_psi_val, k_alpha_val, AE_val

if not par:
    for idx, _ in np.ndenumerate(w_n):
        i, j = idx
        vals = solve_k(idx)
        k_psi[i,j] = vals[0]
        k_alpha[i,j] = vals[1]
        AE[i,j] = vals[2]
    
if par:
    if __name__ == '__main__':
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(solve_k, [idx for idx, _ in np.ndenumerate(w_n)])
        pool.close()
        pool.join()
        for idx, vals in zip([idx for idx, _ in np.ndenumerate(w_n)], results):
            i, j = idx
            k_psi[i,j] = vals[0]
            k_alpha[i,j] = vals[1]
            AE[i,j] = vals[2]


        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)

        lvl_res = 27
        k_psi_lvls = np.linspace(-2.0, 2.0, lvl_res)
        k_alpha_lvls = np.linspace(-15.0, 15.0, lvl_res)
        AE_lvls = np.linspace(0.0, 2.2, lvl_res)

        # use contourf
        c = ax[0].contourf(w_n, w_T, k_psi, levels=k_psi_lvls, cmap='RdBu_r')
        cb = plt.colorbar(c, ax=ax[0], ticks=[k_psi_lvls.min(), k_psi_lvls.max()])
        cb.ax.set_yticklabels([f'{k_psi_lvls.min():.2f}', f'{k_psi_lvls.max():.2f}'])
        ax[0].set_title(r'$\hat{\kappa}_\psi$')
        ax[0].set_xlabel(r'$\hat{\omega}_n$')
        ax[0].set_ylabel(r'$\hat{\omega}_T$')

        c = ax[1].contourf(w_n, w_T, k_alpha, levels=k_alpha_lvls, cmap='RdBu_r', extend='both')
        cb = plt.colorbar(c, ax=ax[1], ticks=[k_alpha_lvls.min(), k_alpha_lvls.max()])
        cb.ax.set_yticklabels([f'{k_alpha_lvls.min():.2f}', f'{k_alpha_lvls.max():.2f}'])
        ax[1].set_title(r'$\hat{\kappa}_\alpha$')
        ax[1].set_xlabel(r'$\hat{\omega}_n$')
        ax[1].set_ylabel(r'$\hat{\omega}_T$')

        c = ax[2].contourf(w_n, w_T, AE, levels=AE_lvls, cmap='gist_heat_r')
        cb = plt.colorbar(c, ax=ax[2], ticks=[AE_lvls.min(), AE_lvls.max()])
        cb.ax.set_yticklabels([f'{AE_lvls.min():.2f}', f'{AE_lvls.max():.2f}'])
        ax[2].set_title(r'$\widehat{A}$')
        ax[2].set_xlabel(r'$\hat{\omega}_n$')
        ax[2].set_ylabel(r'$\hat{\omega}_T$')

        plt.show()