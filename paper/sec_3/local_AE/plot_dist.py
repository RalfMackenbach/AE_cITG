# Here we plot the AE. The plot is saved in the folder plots as AE.png
import source.AE_ITG as ae
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

# enable latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def n_significant_digits_ceil(x,n=1):
    # find the order of magnitude
    order = np.floor(np.log10(np.abs(x)))
    # divide out the order of magnitude
    x = x/10**order
    # round to n significant digits
    x = np.ceil(x*10**n)/10**n
    return x*10**order

def n_significant_digits_floor(x,n=1):
    # find the order of magnitude
    order = np.floor(np.log10(np.abs(x)))
    # divide out the order of magnitude
    x = x/10**order
    # round to n significant digits
    x = np.floor(x*10**n)/10**n
    return x*10**order


res=100
v_par2 =np.geomspace(1e-4,1e1,res)
v_perp2=np.geomspace(1e-4,1e1,res)
# meshgrid for v_par, v_perp
v_par2, v_perp2 = np.meshgrid(v_par2, v_perp2)


w_t = 5.0
w_n = 0.0

w_alpha =-1.0
w_psi   = 1.0

# get k_alpha, k_psi by solve_k
k_psi, k_alpha = ae.solve_k(w_alpha,w_psi,w_n,w_t)

# plot the integrand
ae_integrand = ae.AE_integrand(v_par2,v_perp2,w_alpha,w_psi,w_n,w_t,k_psi,k_alpha) * np.exp(-v_par2-v_perp2)*np.sqrt(v_perp2)

# plot the integrand
plt.figure()
plt.contourf(v_par2,v_perp2,ae_integrand)
plt.xlabel(r'$\hat{v}_{\parallel}^2$')
plt.ylabel(r'$\hat{v}_{\perp}^2$')
plt.colorbar()
plt.title(r'$\widehat{A}$')
# set to log scale
plt.xscale('log')
plt.yscale('log')
plt.savefig('plots/AE_integrand.png',dpi=1000)
plt.show()