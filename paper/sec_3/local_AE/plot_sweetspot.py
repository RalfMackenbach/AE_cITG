# Here we plot the sweet spot where AE is maximal for different values of eta
# (for fixed w_n, w_T, and w_psi=0)

import source.ae as sua
import numpy as np
import scipy.optimize as opt
import multiprocessing as mp
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

eta = np.linspace(2.9, 3.1, 3)


def calculate_AE(w_alpha,eta_val):
    dict =  sua.calculate_AE_arr(3.0, 3.0/eta_val, w_alpha, np.zeros_like(w_alpha))
    return dict['AE'][0]

def find_max_AE(eta_val):
    # find maximal AE for given eta
    w_alpha_init = 1.0
    res = opt.minimize(lambda w_alpha: -calculate_AE(w_alpha,eta_val), w_alpha_init)
    print(f'eta = {eta_val:.2f}, w_alpha = {res.x[0]:+.2f}, AE = {res.fun:+.2f}')
    return res.x[0]



# do the calculation in parallel
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(find_max_AE, eta)
    print(results)
    pool.close()
    pool.join()

    # plot the results
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 8.0/2.4), constrained_layout=True)
    ax.plot(eta, results)
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\omega_{\alpha}$')
    plt.show()
