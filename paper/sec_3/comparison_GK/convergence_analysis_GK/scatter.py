import matplotlib.pyplot as plt
import numpy as np
# enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Paths to the .npz files
npz_files = [
    'scatter_data_GX_start8_end107.npz',
    'scatter_data_GX_start0_end107.npz'
]

fig, axes = plt.subplots(1, 2, figsize=(8*3/4,8*3/4))
markers = ['o','s','^','D','v','<','>','p','h','8','*']

for ax, npz_file in zip(axes, npz_files):
    data = np.load(npz_file)
    keys = list(data.keys())
    # Assume the first key is the x-axis (nominal), rest are y (res)
    x_key = keys[0]
    x = np.log10(data[x_key])
    # Prepare y data for each label
    for i, y_key in enumerate(keys):
        y = np.log10(data[y_key])
        y[y<-1] = -2
        x_plot = x.copy()
        x_plot[x_plot<-1] = -2
        # only keep the label part before the first .
        if '.' in y_key:
            y_key = y_key.split('.')[0]
        label = y_key.replace('_',' ').replace('dbl',r'2 $\cdot$').replace('hlf',r'1/2 $\cdot$')
        label = r'{}'.format(label)
        if label == r'input':
            label = 'nominal'
        ax.scatter(x_plot, y, label=label, s=2, marker=markers[i%len(markers)], alpha=1.0)
    x_line = np.linspace(-1,2,100)
    y_line = x_line
    ax.plot(x_line, y_line, 'k--', label=r'$Q_{\rm nom}= Q_{\rm res}$')
    ax.set_xticks(np.arange(-2,3,1))
    ax.set_yticks(np.arange(-2,3,1))
    ax.set_yticklabels([r'$-\infty$','-1','0','1','2'])
    ax.set_xticklabels([r'$-\infty$','-1','0','1','2'])
    ax.set_xlim(-2.1,2)
    ax.set_ylim(-2.1,2)
    ax.set_aspect('equal', adjustable='box')
    ax.grid()
    # ax.legend(fontsize=8)  # Remove per-axis legend

# y label only on the left-most plot
axes[0].set_ylabel(r'$\log_{10} Q_{\rm res}$')
# x label only on the bottom plots
axes[0].set_xlabel(r'$\log_{10} Q_{\rm nom}$')
axes[1].set_xlabel(r'$\log_{10} Q_{\rm nom}$')

# Single legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.1))

plt.subplots_adjust(wspace=0)
plt.tight_layout()
plt.savefig('a_heat_flux_scatter_side_by_side.png',dpi=1000)
plt.close()
