import source.AE_ITG as ae
import numpy as np
import source.plot_settings as ps
import matplotlib.pyplot as plt
import scipy.integrate

n_levels = 30

# enable latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# calculate the H function
x_min, x_max = -3.0, 3.0
b_min, b_max = -1.0, 1.0
x = np.linspace(x_min, x_max, 10)
b = np.linspace(b_min,b_max,10)

# meshgrid
X, B = np.meshgrid(x,b)
# make container for the result
y = np.zeros_like(X)
y_0 = np.zeros_like(X)

# b is not vectorized, so we need to loop over it
for i in range(len(b)):
    y[i,:] = ae.H_iso(np.exp(x),b[i])


# also do the calculation manually (execute the integral)
integrand = lambda x,b: b**2 * np.exp(-b*x) * ae.I_iso(x)
# define the integral, going from 0 to x
H_int = lambda x,b: scipy.integrate.quad(integrand,0.0,x,args=(b,))[0]


# calculate the integral
H_int_vec = np.vectorize(H_int)
y_int = H_int_vec(X,B)


# plot both the H function and the difference
fig, ax = plt.subplots(1, 2, figsize=(8.0,3.5), constrained_layout=True)


def n_significant_digits(x,n=1):
    # find the order of magnitude
    order = np.floor(np.log10(np.abs(x)))
    # divide out the order of magnitude
    x = x/10**order
    # round to n significant digits
    x = np.ceil(x*10**n)/10**n
    return x*10**order



# plot the H function and the difference
# make levels for the H function, centered around 0
max_val = np.max(np.abs(y))
# find number of significant digits
max_val = n_significant_digits(max_val)
# find the levels
levels = np.linspace(-max_val,max_val,n_levels)


# same for the difference
max_val_diff = np.max(np.abs(y - y_int))
max_val_diff = n_significant_digits(max_val_diff)
# find the levels
levels_diff = np.linspace(0.0,max_val_diff,n_levels)
print(max_val_diff)

# plot the H function and the difference
ax[0].contourf(X, B, y, levels=levels, cmap='bwr')
ax[1].contourf(X, B, np.abs(y - y_int), levels=levels_diff, cmap='gist_heat_r')

# add labels
ax[0].set_xlabel(r'$v_{0}^2$')
ax[0].set_ylabel(r'$b$')

ax[1].set_xlabel(r'$v_{0}^2$')
ax[1].set_ylabel(r'$b$')

# set limits
ax[0].set_xlim(x_min, x_max)
ax[0].set_ylim(b_min, b_max)

ax[1].set_xlim(x_min, x_max)
ax[1].set_ylim(b_min, b_max)

# add colorbar at top of both plots
cbar = fig.colorbar(ax[0].collections[0], ax=ax[0], orientation='horizontal')
cbar.set_label(r'$\mathcal{H}(v_0^2,b)$')
# set ticks at -max_val, 0, max_val
cbar.set_ticks([-max_val, 0, max_val])

cbar = fig.colorbar(ax[1].collections[0], ax=ax[1], orientation='horizontal')
cbar.set_label(r'$\left|\mathcal{H}(v_0^2,b) - \mathcal{H}_{\rm num}(v_0^2,b) \right|$')
# set ticks at 0, max_val_diff
cbar.set_ticks([0, max_val_diff])

# save the plot
plt.savefig('plots/H_diff.png', dpi=ps.dpi)

plt.show()

# # plot the H function difference
# ax.contourf(X, B, np.abs(y-y_int), levels=21, cmap='gist_heat_r')

# # add labels
# ax.set_xlabel(r'$v_{0}^2$')
# ax.set_ylabel(r'$b$')

# # set limits
# ax.set_xlim(x_min, x_max)
# ax.set_ylim(b_min, b_max)

# # add colorbar
# cbar = fig.colorbar(ax.collections[0], ax=ax, orientation='vertical')
# cbar.set_label(r'$\left|\mathcal{H}(v_0^2,b) - \mathcal{H}(0,b) - \mathcal{H}_{\rm num}(v_0^2,b) \right|$')

# # save the plot
# plt.savefig('plots/H_diff.png', dpi=ps.dpi)

# plt.show()
