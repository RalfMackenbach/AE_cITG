import source.ae as ae
import numpy as np
import matplotlib.pyplot as plt
# enable LaTeX font rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# here we check whether the strong gradient limit is working

# make array of w_alpha, w_psi (of length 1)
w_alpha = np.array([1.0])
w_psi = np.array([1.0])
w_T = np.geomspace(1e-2,1e5,100)
eta_inv = 0.5
w_n = w_T * eta_inv

# make container for results
AE_strong = np.ones_like(w_n)
AE_full = np.zeros_like(w_n)

# calculate AE in the strong gradient limit
for i in range(len(w_n)):
    print(f"Calculating AE for w_T = {w_T[i]}")
    dict = ae.calculate_AE_arr(w_T=w_T[i], w_n=w_n[i], w_alpha=w_alpha, w_psi=w_psi)
    AE_full[i] = dict['AE'][0]

dict_strong = ae.calculate_AE_strong_arr(w_T=1.0, w_n=1.0*eta_inv, w_alpha=w_alpha, w_psi=w_psi)
AE_strong = AE_strong * dict_strong['AE'][0] * w_T

# plot AE_strong and AE_full on left y-axis, relative error on right y-axis
fac = 3/4
fig, ax1 = plt.subplots(figsize=(fac * 8/2, fac * 8/2.5), constrained_layout=True)

color1 = 'tab:blue'
color2 = 'tab:orange'
color3 = 'tab:green'
# Create a second y-axis for the relative error
ax2 = ax1.twinx()
rel_error = np.abs(AE_strong - AE_full) / AE_full
ax2.plot(w_T, rel_error, color='red', linestyle=':')
ax2.set_ylabel(r'$|\widehat{A}_{\rm strong} - \widehat{A}|/\widehat{A}$', color='red')
ax2.tick_params(axis='y', colors='red')
ax2.set_yscale('log')

# Plot AE_strong and AE_full
ax1.plot(w_T, AE_full, label=r'$\widehat{A}$', color=color2, linestyle='-')
ax1.plot(w_T, AE_strong, label=r'$\widehat{A}_{\rm strong}$', color=color1, linestyle='--')
ax1.set_xlabel(r'$\hat{\omega}_T$')
ax1.set_ylabel(r'$\widehat{A}$')
ax1.set_xscale('log')
ax1.set_yscale('log')


# add legend
ax1.legend(loc='upper center')


# save figure
plt.savefig('AE_strong_vs_full.png', dpi=1000)