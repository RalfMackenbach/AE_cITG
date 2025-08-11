# adjusted from test_ae.py, serves as an example of how to run the AE functions with arrays
import numpy as np
import source.ae as ae

# make array of w_alpha, w_psi (s-alpha like)
s = -1.0 # magnetic shear
npol = 1.0 # number of poloidal turns
z = np.linspace(-npol*np.pi,npol*np.pi,48) # arclength, parallel coordinate
w_psi = np.sin(z) # example w_psi, simple sine function
w_alpha = -np.cos(z) - s * z * w_psi # example w_alpha dependent on w_psi and shear
B = 1.0 - 0.1 * np.cos(z)  # example sinusoidal magnetic field


dlnpdx = -100.0 # dlnp/dx = dlnn/dx + dlnT/dx
eta  = 2.0 # ratio of temperature gradient to density gradient , dlnT/dx = eta * dlnp/dx
# invert to find w_T and w_n
w_T = -((dlnpdx * eta)/(1 + eta))
w_n = -dlnpdx / (1 + eta)  # normalized frequency for temperature and density
print(f"w_T: {w_T}, w_n: {w_n}")



ans = ae.calculate_AE_arr(w_T=w_T, w_n=w_n, w_alpha=w_alpha, w_psi=w_psi)

# calculate the available energy 
available_energy = np.mean(ans['AE']/B)  / np.mean(1/B) 
print(f"Available Energy: {available_energy}")

# plot intput and output in 2X3 grid
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False  # Set to True if you have LaTeX installed

fig, axs = plt.subplots(2,3, figsize=(15/2,10/2), sharex=True, constrained_layout=True)
axs[0,0].plot(z,w_alpha)
axs[0,0].set_title(r'$\omega_\alpha$')
axs[0,1].plot(z,w_psi)
axs[0,1].set_title(r'$\omega_\psi$')
axs[0,2].plot(z,ans['AE'])
axs[0,2].set_title(r'$\mathrm{AE}$')
axs[1,0].plot(z,ans['k_alpha'])
axs[1,0].set_title(r'$\kappa_\alpha$')
axs[1,1].plot(z,ans['k_psi'])
axs[1,1].set_title(r'$\kappa_\psi$')
axs[1,2].plot(z,B)
axs[1,2].set_title(r'$B$')
plt.xlim(z.min(),z.max())
plt.show()