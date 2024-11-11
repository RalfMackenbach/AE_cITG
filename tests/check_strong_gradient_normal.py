import numpy as np
import matplotlib.pyplot as plt
import source.ae_ITG as ae

gradient_strength   = np.geomspace(1,1000,100)
w_alpha             =-1.0
w_psi               = 1.0
w_n                 = 1.0*gradient_strength
w_T                 = 3.0*gradient_strength

AEs = np.zeros_like(gradient_strength)
AEs_strong = np.zeros_like(gradient_strength)

for idx, _ in np.ndenumerate(gradient_strength):
    idx = idx[0]
    AE_dict = ae.calculate_AE(w_alpha,w_psi,w_n[idx],w_T[idx])
    AE_strong_dict = ae.calculate_AE_strong(w_alpha,w_psi,w_n[idx],w_T[idx])
    AEs[idx] = AE_dict['AE']
    AEs_strong[idx] = AE_strong_dict['AE']

# plot both
plt.figure()
plt.plot(gradient_strength,AEs,label='Full calculation')
plt.plot(gradient_strength,AEs_strong,label='Strong gradient approximation')
plt.plot(gradient_strength,np.abs(AEs-AEs_strong)/AEs,label='Relative difference')
plt.xlabel('Gradient strength')
plt.ylabel('Available energy')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.show()