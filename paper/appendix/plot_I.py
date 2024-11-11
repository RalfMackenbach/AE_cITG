import source.AE_ITG as ae
import numpy as np
import source.plot_settings as ps
import matplotlib.pyplot as plt

# enable latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(1, 1, figsize=(3.0,2.5), constrained_layout=True)

# calculate the I function
x_min, x_max = -3, 3
x = np.linspace(x_min, x_max, 1000)
y = ae.I_iso(x)

# plot the I function
ax.plot(x, y)

# add labels
ax.set_xlabel(r'$v_{0}^2$')
ax.set_ylabel(r'$\mathcal{I}$')

# set limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(bottom=0)

# add grid
ax.grid(True)

# save the plot
plt.savefig('plots/I.png', dpi=ps.dpi)

plt.show()