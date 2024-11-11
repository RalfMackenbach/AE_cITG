# import all necessary packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib as mpl

# enable latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# load the data
k_psi = np.load('data/k_psi.npy')
k_psi_strong = np.load('data/k_psi_strong.npy')
k_psi_iso = np.load('data/k_psi_iso.npy')

# w_T and w_n go from -10 to 10
w_n = np.linspace(-10.0,10.0,100)
w_T = np.linspace(-10.0,10.0,100)
w_n, w_T = np.meshgrid(w_n,w_T)

# make a figure
scale_fac=3/4
fig, ax = plt.subplots(3,3,figsize=(scale_fac*8.0,scale_fac*8.0),constrained_layout=True)

# find max values of all AE
max_val = np.nanmax([np.nanmax(k_psi),np.nanmax(k_psi_strong),np.nanmax(k_psi_iso)])
min_val = np.nanmin([np.nanmin(k_psi),np.nanmin(k_psi_strong),np.nanmin(k_psi_iso)])
print(min_val,max_val)
# override and set to -10,10
min_val = -2.2
max_val = 2.2
levels = np.linspace(min_val,max_val,26)

# make plots. Rows correspond to AE, AE_strong, AE_iso. Columns correspond cases of w_alpha, w_psi
# first plot AE. cmap is gist_heat_t
cmap = plt.get_cmap('bwr')

# plot AE
for i in range(3):
    ax[0,i].contourf(w_n,w_T,k_psi[:,:,i],levels=levels,cmap=cmap,extend='both')

# plot AE_strong
for i in range(3):
    ax[1,i].contourf(w_n,w_T,k_psi_strong[:,:,i],levels=levels,cmap=cmap,extend='both')

# plot AE_iso
for i in range(3):
    ax[2,i].contourf(w_n,w_T,k_psi_iso[:,:,i],levels=levels,cmap=cmap,extend='both')

# set labels, only on the left and bottom plots
ax[2,0].set_xlabel(r'$\hat{\omega}_n$')
ax[2,1].set_xlabel(r'$\hat{\omega}_n$')
ax[2,2].set_xlabel(r'$\hat{\omega}_n$')
ax[0,0].set_ylabel(r'$\hat{\omega}_T$')
ax[1,0].set_ylabel(r'$\hat{\omega}_T$')
ax[2,0].set_ylabel(r'$\hat{\omega}_T$')

# add colorbars to rows, to the right of the rows. labels are \widehat{A}, \widehat{A}_{\text{strong}}, \widehat{A}_{\text{iso}}
# extend='both' adds arrows to the colorbar
cbar1 = fig.colorbar(ax[0,2].contourf(w_n,w_T,k_psi[:,:,2],levels=levels,cmap=cmap),ax=ax[0,2])
cbar2 = fig.colorbar(ax[1,2].contourf(w_n,w_T,k_psi_strong[:,:,2],levels=levels,cmap=cmap),ax=ax[1,2])
cbar3 = fig.colorbar(ax[2,2].contourf(w_n,w_T,k_psi_iso[:,:,2],levels=levels,cmap=cmap),ax=ax[2,2])
cbar1.set_label(r'$\hat{\kappa}_{\psi}$')
cbar2.set_label(r'$\hat{\kappa}_{\psi, {\rm strong}}$')
cbar3.set_label(r'$\hat{\kappa}_{\psi, {\rm iso}}$')
cbar1.set_ticks([min_val,max_val])
cbar2.set_ticks([min_val,max_val])
cbar3.set_ticks([min_val,max_val])
# set titles on top of rows (0.0,1.0),(-1.0,0.0),(-1.0,1.0)
ax[0,0].set_title(r'$(\hat{\omega}_{\alpha},\hat{\omega}_{\psi}) = (0,1)$')
ax[0,1].set_title(r'$(\hat{\omega}_{\alpha},\hat{\omega}_{\psi}) = (-1,0)$')
ax[0,2].set_title(r'$(\hat{\omega}_{\alpha},\hat{\omega}_{\psi}) = (-1,1)$')

# add text (a), (b), (c), ... to the plots, from top left to bottom right
text_arr = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
for i in range(3):
    for j in range(3):
        # be sure to center the text at given position
        ax[i,j].text(-7.,-8.,text_arr[3*i+j],fontsize=12,color='black',va='center',ha='center')

plt.savefig('k_psi_all.png',dpi=1000)
plt.show()
