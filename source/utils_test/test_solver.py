import numpy as np
import source.utils_solver as sus
import time
import matplotlib.pyplot as plt

def mapping(x,theta=3.0):
    # rotation and translation
    x = np.asarray(x)
    return np.array([(x[0]*np.cos(theta) - x[1]*np.sin(theta))/1.1,
                     (x[0]*np.sin(theta) + x[1]*np.cos(theta))/3.0])

true_val = np.array([0.0, 0.0])
x0 = np.array([np.exp(1), np.pi])

# define methods
methods = np.asarray(['iterative', 'fsolve', 'least_squares', 'newton_krylov', 'broyden1', 'broyden2'])

results = np.zeros((len(methods), 2))
times = np.zeros((len(methods), 2))

for i, method in enumerate(methods):
    start = time.time()
    results[i] = sus.solver(mapping, x0, method=method, abs_tol=1e-8, rel_tol=1e-8)
    times[i] = time.time() - start

print(results)

# plot the results
fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
for i, method in enumerate(methods):
    ax[0].bar(method, np.linalg.norm(results[i] - true_val))
    ax[1].bar(method, times[i])
ax[0].set_ylabel('Norm of error')
ax[1].set_ylabel('Time (s)')
# set to log scale
ax[0].set_yscale('log')
ax[1].set_yscale('log')
# rotate the x-axis labels by 90 degrees
plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=90)
plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=90)

plt.show()