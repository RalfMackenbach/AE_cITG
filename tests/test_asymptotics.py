import source.ae as ae
import numpy as np
import matplotlib.pyplot as plt

# here we check whether the strong gradient limit is working

# make array of w_alpha, w_psi (of length 1)
w_alpha = np.array([1.0])
w_psi = np.array([1.0])
w_T = np.geomspace(1e-2,1e5,100)
eta_inv = 0.1
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

# plot the results
# 1x2, loglog, left values, right relative error
fig, axs = plt.subplots(1,2, figsize=(15/2,10/2), sharex=True, constrained_layout=True)
axs[0].plot(w_T,AE_strong,label='Strong gradient limit')
axs[0].plot(w_T,AE_full,label='Regular')
axs[0].set_xlabel(r'$\hat{\omega}_T$')
axs[0].set_ylabel(r'$\widehat{A}$')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].legend()

# calculate relative error
rel_error = np.abs(AE_strong - AE_full) / AE_strong
axs[1].plot(w_T,rel_error)
axs[1].set_xlabel(r'$w_n$')
axs[1].set_ylabel(r'$\frac{|AE_{strong} - AE|}{AE}$')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
plt.show()


# import matplotlib.pyplot as plt
# plt.figure(figsize=(6,4))
# plt.plot(w_n,AE_strong,label='Strong gradient limit')
# plt.plot(w_n,AE,label='Regular')
# plt.xlabel(r'$w_n$')
# plt.ylabel(r'$AE$')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.show()