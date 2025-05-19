import source.ae as sua
import source.utils_ae_full as suaf
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# make grid of omega_alpha and omega_psi [-10,10]x[-10,10]
res = 1000
lims = 4.0
vperp   = np.linspace( 0.0, lims, res)
vpar    = np.linspace(-lims, lims, res)
# meshgrid
vperp, vpar = np.meshgrid(vperp, vpar)

# give values to omega_n and omega_T
w_n = 3.0/3.0
w_T = 1.0/3.0

# Define different values for w_alpha and w_psi
w_alpha_values  = np.asarray([-2.0, -0.5, 0.5, 2.0])
w_psi_values    = np.asarray([1.0, 1.0, 1.0, 1.0])/2

def calculate_AE(w_T, w_n, w_alpha, w_psi):
    dict = sua.calculate_AE_arr(w_T, w_n, np.asarray([w_alpha]), np.asarray([w_psi]))
    return dict

# make four dicts
dicts = []
for idx, (w_alpha, w_psi) in enumerate(zip(w_alpha_values, w_psi_values)):
    dict = calculate_AE(w_T, w_n, w_alpha, w_psi)
    dicts.append(dict)

arrs = []


# Create a 2x2 plot
fac = 3 / 4
fig, axes = plt.subplots(2, 2, figsize=(fac * 8, fac * 8/1.8), constrained_layout=True, sharex=True, sharey=True)



for i, (w_alpha, w_psi) in enumerate(zip(w_alpha_values, w_psi_values)):
    ans = dicts[i]
    k_psi = ans['k_psi'][0]
    k_alpha = ans['k_alpha'][0]

    arr = suaf.AE_integrand(vperp**2, vpar**2, w_alpha, w_psi, w_n, w_T, k_psi, k_alpha) * 1.0 / (6 * np.sqrt(np.pi)) * np.exp(-vpar**2 - vperp**2)
    arrs.append(arr)

# make individual colorbars for each plot with min=0 and max=arr.max() for that plot
for i, (w_alpha, w_psi) in enumerate(zip(w_alpha_values, w_psi_values)):
    ax = axes[i // 2, i % 2]
    arr_map = arrs[i]
    max_val = np.max(arr_map)
    min_val = 0.0
    levels = np.linspace(min_val, max_val, 20)
    contour = ax.contourf(vpar, vperp, arr_map, cmap='gist_heat_r', levels=levels)
    ax.set_title(r'$(\hat{\omega}_\alpha, \hat{\omega}_\psi) = $' + rf'$({w_alpha}, {w_psi})$')
    ax.set_xlabel(r'$\hat{v}_{\|}$')
    ax.set_ylabel(r'$\hat{v}_{\perp}$')
    # Add a colorbar for each subplot
    cbar = fig.colorbar(contour, ax=ax, label=r'$\log_{10} \widehat{\mathcal{A}}$' if i % 2 == 1 else None)
    cbar.set_ticks([min_val, max_val])
    cbar.ax.set_yticklabels([f'{min_val:.1e}', f'{max_val:.1e}'])

# add title
plt.suptitle(r'$(\hat{\omega}_n, \hat{\omega}_T) = (1,1/3)$')

# # set the aspect ratio to be equal
# for ax in axes.flatten():
#     ax.set_aspect('equal')

# Save the figure
plt.savefig('plots/AE_integrand_small_eta.png', dpi=1000)

plt.show()
