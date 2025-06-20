# This script plots the gradients of the nonlinear gyrokinetic simulations for the random-gradient subset.

import numpy as np
import matplotlib.pyplot as plt
# enable latex rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def gen_prims(idx):
    # set random seed
    if idx is not None:
        np.random.seed(idx)
    eta_angle_std   = np.arctan2(3.0, 0.9)
    dpdx_std        = 3.9
    angle_min       = np.arctan2(1.0, 3.0)
    # generate random values centered at std values
    while True:
        eta_angle = np.random.normal(eta_angle_std, 0.25)
        dpdx_rand = np.random.normal(dpdx_std, 1.0)
        angle_rand = np.arctan2(np.sin(eta_angle), np.cos(eta_angle))
        if dpdx_rand >= 0.5 and angle_min <= angle_rand <= (np.pi / 2 + 0.1):
                break
    # generate random int [0,1,2]
    p0, p1, p2 = 1/3, 1/3, 1/3  # probabilities for each case
    rand_int = np.random.choice([0, 1, 2], p=[p0, p1, p2])
    # if zero change both dpdx and eta
    if rand_int == 0:
        dpdx = dpdx_rand
        eta_angle = angle_rand
    # if one change only dpdx
    elif rand_int == 1:
        dpdx = dpdx_rand
        eta_angle = eta_angle_std
    # if two change only eta
    elif rand_int == 2:
        dpdx = dpdx_std
        eta_angle = angle_rand
    # calculate tprim and fprim
    tprim_unit = np.sin(eta_angle)
    fprim_unit = np.cos(eta_angle)
    # (tprim_unit + fprim_unit)*C = dpdx -> C = dpdx / (tprim_unit + fprim_unit)
    C = dpdx / (tprim_unit + fprim_unit)
    tprim = tprim_unit * C
    fprim = fprim_unit * C
    # if within delta of std fprim and tprim, resample
    delta = 0.1
    radius = np.sqrt((tprim - 3.0) ** 2 + (fprim - 0.9) ** 2)
    if radius < delta:
        return gen_prims(None)
    return tprim, fprim



# calculate eta for a 30001 random indices and plot histogram
angles = []
tprims = []
fprims = []
for i in range(0,30001):
    print(i)
    tprim, fprim = gen_prims(i)
    eta = tprim / fprim
    # get angle
    angle = np.arctan2(tprim, fprim)
    angles.append(angle)
    tprims.append(tprim)
    fprims.append(fprim)
plt.hist(angles, bins=100,log=True)
# add eta=2/3 line
plt.axvline(np.arctan2(2.0,3.0), color='r', linestyle='dashed', linewidth=2)
# count what % of angles have eta<2/3
count = 0
for angle in angles:
    if angle < np.arctan2(2.0,3.0):
        count += 1
print('Percentage of angles with eta < 2/3: ', count/len(angles)*100)
plt.xlabel('angle')
plt.ylabel('count')
plt.show()

# also scatter plot tprim vs fprim using fig and ax
scale=3/4/2
fig, ax = plt.subplots(figsize=(scale*8,scale*8),tight_layout=True)
ax.scatter(fprims, tprims, s=0.3, edgecolors='none')
# add eta=2/3 line
w_ns = np.linspace(-1, 6, 100)
# ax.plot(w_ns, 2.0/3.0*w_ns, 'r')
ax.set_ylim(0, 8)
ax.set_xlim(-1, 4)
ax.set_xlabel(r'$-\partial_\rho \ln n$')
ax.set_ylabel(r'$-\partial_\rho \ln T$')

# save figure
fig.savefig('plots/tprim_vs_fprim.png', dpi=1000)

plt.show()