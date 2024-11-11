import source.AE_ITG as ae
import numpy as np
import source.plot_settings as ps
import matplotlib.pyplot as plt

# enable latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(1, 1, figsize=(3.0,2.5), constrained_layout=True)

# calculate the J function
x_min, x_max = -3, 3
x = np.linspace(x_min, x_max, 1000)
y1 = ae.J_iso(x, 1.0)
y2 = ae.J_iso(x, -1.0)

# plot the J function
ax.plot(x, y1, label=r'$\Omega=+1$')
ax.plot(x, y2, label=r'$\Omega=-1$')

# add labels
ax.set_xlabel(r'$v_{0}^2$')
ax.set_ylabel(r'$\mathcal{J}(v_0^2,\Omega)$')

# add legend
ax.legend()

# set limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(bottom=0)

# add grid
ax.grid(True)

# save the plot
plt.savefig('plots/J.png', dpi=ps.dpi)

plt.show()