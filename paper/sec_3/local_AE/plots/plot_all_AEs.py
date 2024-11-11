# here we import all data from /data (AE_iso.npy, AE_strong.npy, AE.npy) and plot them in a 3X3 grid

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib as mpl

# enable latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# load the data
AE = np.load('data/AE.npy')
AE_strong = np.load('data/AE_strong.npy')
AE_iso = np.load('data/AE_iso.npy')

# w_T and w_n go from -10 to 10
w_n = np.linspace(-10.0,10.0,100)
w_T = np.linspace(-10.0,10.0,100)
w_n, w_T = np.meshgrid(w_n,w_T)

# make a figure
scale_fac=3/4
fig, ax = plt.subplots(3,3,figsize=(scale_fac*8.0,scale_fac*7.0),constrained_layout=True)

# find max values of all AE
max_val = 2.2 #np.max([np.max(AE),np.max(AE_strong),np.max(AE_iso)])
levels = np.linspace(0,max_val,25)

# make plots. Rows correspond to AE, AE_strong, AE_iso. Columns correspond cases of w_alpha, w_psi
# first plot AE. cmap is gist_heat_t
cmap = plt.get_cmap('gist_heat_r')

# plot AE
for i in range(3):
    ax[0,i].contourf(w_n,w_T,AE[:,:,i],levels=levels,cmap=cmap)

# plot AE_strong
for i in range(3):
    ax[1,i].contourf(w_n,w_T,AE_strong[:,:,i],levels=levels,cmap=cmap)

# plot AE_iso
for i in range(3):
    ax[2,i].contourf(w_n,w_T,AE_iso[:,:,i],levels=levels,cmap=cmap)

# set labels, only on the left and bottom plots
ax[2,0].set_xlabel(r'$\hat{\omega}_n$')
ax[2,1].set_xlabel(r'$\hat{\omega}_n$')
ax[2,2].set_xlabel(r'$\hat{\omega}_n$')
ax[0,0].set_ylabel(r'$\hat{\omega}_T$')
ax[1,0].set_ylabel(r'$\hat{\omega}_T$')
ax[2,0].set_ylabel(r'$\hat{\omega}_T$')

# add colorbars to rows, to the right of the rows. labels are \widehat{A}, \widehat{A}_{\text{strong}}, \widehat{A}_{\text{iso}}
cbar1 = fig.colorbar(ax[0,2].contourf(w_n,w_T,AE[:,:,2],levels=levels,cmap=cmap),ax=ax[0,2])
cbar2 = fig.colorbar(ax[1,2].contourf(w_n,w_T,AE_strong[:,:,2],levels=levels,cmap=cmap),ax=ax[1,2])
cbar3 = fig.colorbar(ax[2,2].contourf(w_n,w_T,AE_iso[:,:,2],levels=levels,cmap=cmap),ax=ax[2,2])
cbar1.set_label(r'$\widehat{A}$')
cbar2.set_label(r'$\widehat{A}_{\rm strong}$')
cbar3.set_label(r'$\widehat{A}_{\rm iso}$')
cbar1.set_ticks([0,max_val])
cbar2.set_ticks([0,max_val])
cbar3.set_ticks([0,max_val])
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



plt.savefig('AE_all.png',dpi=1000)
plt.show()
