# plot the scatter
import matplotlib.pyplot as plt
import numpy as np
from IO import *

# enable latex rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# load Q data
data_1 = load_data('20240601-01-103_gx_nfp4_production_gx_results_gradxRemoved.pkl')
data_2 = load_data('20240726-01-random_stellarator_equilibria_and_GX_gx_results_gradxRemoved.pkl')
data_3 = load_data('20241004-01-random_stellarator_equilibria_GX_results_combined.pkl')

# tube data
tube_data_1 = load_data('20240601-01-assembleFluxTubeMatrix_noShiftScale_nz96_withCvdrift.pkl')
tube_data_2 = load_data('20240726-01-assembleFluxTubeTensor_vacuum_nz96.pkl')
tube_data_3 = load_data('20241004-01-assembleFluxTubeTensor_multiNfp_finiteBeta_nz96.pkl')
print(tube_data_1.keys())
tube_names = np.asarray(tube_data_3['tube_files'])
# find indices of tube files with different Nfp
idx_nfp2 = np.asarray([i for i in range(len(tube_names)) if 'nfp2' in tube_names[i]])
idx_nfp3 = np.asarray([i for i in range(len(tube_names)) if 'nfp3' in tube_names[i]])
idx_nfp4 = np.asarray([i for i in range(len(tube_names)) if 'nfp4' in tube_names[i]])
idx_nfp5 = np.asarray([i for i in range(len(tube_names)) if 'nfp5' in tube_names[i]])


# load AE data
file_path = "AE_processed_data/"
AE_1 = np.load(file_path+'20240601-01-assembleFluxTubeMatrix_noShiftScale_nz96_withCvdrift_AE.npy').flatten()
AE_2 = np.load(file_path+'20240726-01-assembleFluxTubeTensor_vacuum_nz96_AE.npy').flatten()
AE_3 = np.load(file_path+'20241004-01-assembleFluxTubeTensor_multiNfp_finiteBeta_nz96_AE.npy').flatten()

# get the dictionary
Q_1 = data_1['Q_avgs_without_FSA_grad_x'].flatten()
Q_2 = data_2['Q_avgs_without_FSA_grad_x'].flatten()
Q_3 = data_3['Q_avgs_without_FSA_grad_x'].flatten()

# scatter plot, split by Nfp. Q1 and Q2 are nfp4, Q3 is nfp2,3,4,5
# labels
nfp4_AE = np.concatenate((AE_1,AE_2,AE_3[idx_nfp4])).flatten()
nfp4_Q = np.concatenate((Q_1,Q_2,Q_3[idx_nfp4])).flatten()
nfp3_AE = (AE_3[idx_nfp3]).flatten()
nfp3_Q = (Q_3[idx_nfp3]).flatten()
nfp2_AE = (AE_3[idx_nfp2]).flatten()
nfp2_Q = (Q_3[idx_nfp2]).flatten()
nfp5_AE = (AE_3[idx_nfp5]).flatten()
nfp5_Q = (Q_3[idx_nfp5]).flatten()

# plot the scatter. Many points (>10^5) so use alpha to make it more visible
fig, ax = plt.subplots(1,2,constrained_layout=True,figsize=(8,4))
s_val=1
alpha_val=0.05
marker_shape='o'
ax[0].scatter(nfp4_AE,nfp4_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:blue')
ax[0].scatter(nfp3_AE,nfp3_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:orange')
ax[0].scatter(nfp2_AE,nfp2_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:green')
ax[0].scatter(nfp5_AE,nfp5_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:red')
ax[0].set_xlabel(r'$\widehat{A}$')
ax[0].set_ylabel(r'$\widehat{Q}$')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
# same thing for second plot, but with limits
ax[1].scatter(nfp4_AE,nfp4_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:blue')
ax[1].scatter(nfp3_AE,nfp3_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:orange')
ax[1].scatter(nfp2_AE,nfp2_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:green')
ax[1].scatter(nfp5_AE,nfp5_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:red')
ax[1].set_xlabel(r'$\widehat{A}$')
ax[1].set_ylabel(r'$\widehat{Q}$')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlim([2e-3,5e-1])
ax[1].set_ylim([1e-3,1e3])

# fit a 3/2 power law to the data
AEs = np.asarray(np.concatenate((nfp4_AE,nfp3_AE,nfp2_AE,nfp5_AE))).flatten()
Qs = np.asarray(np.concatenate((nfp4_Q,nfp3_Q,nfp2_Q,nfp5_Q))).flatten()
fit_func = lambda logx, loga: loga + 3/2*logx
# fit the data
from scipy.optimize import curve_fit
popt, pcov = curve_fit(fit_func,np.log(AEs),np.log(Qs))
# plot the fit
x_fit = np.linspace(1e-3,1e0,100)
y_fit = np.exp(fit_func(np.log(x_fit),*popt))
ax[1].plot(x_fit,y_fit,'k--',label=r'$Q \propto A^{3/2}$')


# add legend manually with alpha=1
ax[1].scatter([],[],label=r'$N_{fp}=2$',alpha=1,s=5,marker=marker_shape,color='tab:green')
ax[1].scatter([],[],label=r'$N_{fp}=3$',alpha=1,s=5,marker=marker_shape,color='tab:orange')
ax[1].scatter([],[],label=r'$N_{fp}=4$',alpha=1,s=5,marker=marker_shape,color='tab:blue')
ax[1].scatter([],[],label=r'$N_{fp}=5$',alpha=1,s=5,marker=marker_shape,color='tab:red')
ax[1].legend(loc='upper right',fontsize=8)


# save the figure
plt.savefig('scatter_AE_Q.png',dpi=1000)

plt.show()