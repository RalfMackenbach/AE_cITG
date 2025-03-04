import numpy as np
import source.utils_integral as sui
import time
import matplotlib.pyplot as plt

a = 1.0
b = 1.0
c = 1.0
d = 1.0

def f(x, y):
    # check if x and y have the same shape
    x = np.asarray(x)
    y = np.asarray(y)
    return np.ones_like(y) + a*x + b*y + c*np.sin(y) + d*np.cos(x)
true_val = 1/2 * (2 + 2 * a + b + d + 2**(3/4) * c * np.sin(np.pi/8)) *np.sqrt(np.pi)


# define methods and number of quadrature points
methods = np.asarray(['dblquad','dblquad_uv', 'dbltanhsinh_uv','gauss_laguerre',
                      'hermite_laguerre', 'gaussian','midpoint', 'trapz', 'simpson', 
                      'boole', 'clenshaw_curtis','cheb_gauss','gauss_jacobi'])
n_quad_values = np.geomspace(10, 1000, 20, dtype=int)

results = np.zeros((len(methods), len(n_quad_values)))
times = np.zeros((len(methods), len(n_quad_values)))

for i, method in enumerate(methods):
    for j, n_quad in enumerate(n_quad_values):
        start = time.time()
        results[i, j] = sui.integrate_2D_AE_weighting(f, method=method, n_quad=n_quad, abs_tol=1.0/n_quad, rel_tol=1.0/n_quad)
        times[i, j] = time.time() - start

# plot the results in a 1x2 grid
linestyles = ['-', '--', '-.', ':']
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for i, method in enumerate(methods):
    ax[0].plot(n_quad_values, np.abs(results[i] - true_val)/true_val, label=method, linestyle=linestyles[i%4])
    ax[1].plot(n_quad_values, times[i], label=method, linestyle=linestyles[i%4])
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('Number of quadrature points')
ax[0].set_ylabel('Relative error')
ax[0].legend()
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel('Number of quadrature points')
ax[1].set_ylabel('Time (s)')
ax[1].legend()
plt.show()