import numpy as np
import matplotlib.pyplot as plt
import source.plot_settings as ps

# enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



fig, ax = plt.subplots(1, 1, figsize=(3.0,2.5), constrained_layout=True) 

# plot the stability region
w_n = np.linspace(-1,1,100)
w_T = np.linspace(-1,1,100)
w_alpha = -1/4
w_intersec = -3/2*w_alpha
w_n_max = np.max(w_n)
w_n_min = np.min(w_n)
ax.plot([w_intersec,w_n_max],[-w_alpha,-w_alpha],'k')
ax.plot([w_intersec,w_n_max],[2/3*w_intersec,2/3*w_n_max],'k')
ax.plot([w_n_min,0],[0,0],'k')
ax.plot([w_n_min,0],[2/3*w_n_min,0],'k')
ax.plot([0,w_n_max],[0,0],'k--')
# add dotted line between w_T = 0 and w_T = 2/3*w_n
ax.plot([0,w_intersec],[0,2/3*w_intersec],linestyle=':',color='k')

# add labels
ax.set_xlabel(r'$- \partial_\psi \ln n$')
ax.set_ylabel(r'$- \partial_\psi \ln T$')

# set limits
ax.set_xlim(np.min(w_n), np.max(w_n))
ax.set_ylim(np.min(w_T), np.max(w_T))

# keep only the zero ticks, point them inwards, but keep the labels outside
ax.set_xticks([0])
ax.set_yticks([0])



# add text in stable region, centered at given position
c=1.3
ax.text(c*-0.6, c*-0.2, 'stable', ha='center', va='center')
ax.text(c*0.61, c*0.2+0.1, 'stable', ha='center', va='center')
# add text stating $\partial_\psi \ln B < 0$
ax.text(0.6, 0.125, r'$\eta < \eta_B$', ha='center', va='center')

# keep aspect ratio fixed
ax.set_aspect('equal')

# save the plot
plt.savefig('plots/stability_diagram.png', dpi=ps.dpi)

# show the plot
plt.show()