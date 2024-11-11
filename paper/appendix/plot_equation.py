import source.AE_ITG as ae
import numpy as np
import source.plot_settings as ps
import matplotlib.pyplot as plt
import scipy.integrate

# enable latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(1, 1, figsize=(4.0,2.5), constrained_layout=True)


# calculate the equation difference
y = np.linspace(1/10,10,100)

w_alpha = -0.1
w_n = 0.1
w_T = 0.6

# loop over y
vals = []
for y_val in y:
    val=ae.equation_tilde_k_alpha_iso_II(y_val, w_alpha, w_n, w_T)
    vals.append(val)

# plot the equation difference
ax.plot(y, vals,label=r'$\tilde{k}_{\alpha} - \tilde{k}_{\alpha}^{(0)}$')
plt.show()