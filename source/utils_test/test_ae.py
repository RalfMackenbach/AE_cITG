# here we test the high-level ae functions
import numpy as np
import source.ae as ae

# make array of w_alpha, w_psi
s = -1.0
npol = 1.0
z = np.linspace(-npol*np.pi,npol*np.pi,201)
w_psi = np.sin(z)
w_alpha = -np.cos(z) - s * z * w_psi

scale = 100.0

ans = ae.calculate_AE_arr(w_T=scale*3.0, w_n=scale*0.9, w_alpha=w_alpha, w_psi=w_psi)

print(np.mean(ans['AE']))

# plot intput and output in 2X3 grid
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,3, figsize=(15/2,10/2), sharex=True, constrained_layout=True)
axs[0,0].plot(z,w_alpha)
axs[0,0].set_title('w_alpha')
axs[0,1].plot(z,w_psi)
axs[0,1].set_title('w_psi')
axs[0,2].plot(z,ans['AE'])
axs[0,2].set_title('AE')
axs[1,0].plot(z,ans['k_alpha'])
axs[1,0].set_title('k_alpha')
axs[1,1].plot(z,ans['k_psi'])
axs[1,1].set_title('k_psi')
plt.xlim(z.min(),z.max())
plt.show()