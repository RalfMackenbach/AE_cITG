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
data_4 = load_data('20241103-01-random_stellarator_equilibria_finiteBeta_randomAspect_allNFP_GX_results_combined.pkl')
# tube data
tube_data_1 = load_data('20240601-01-assembleFluxTubeMatrix_noShiftScale_nz96_withCvdrift.pkl')
tube_data_2 = load_data('20240726-01-assembleFluxTubeTensor_vacuum_nz96.pkl')
tube_data_3 = load_data('20241004-01-assembleFluxTubeTensor_multiNfp_finiteBeta_nz96.pkl')
tube_data_4 = load_data('20241103-01-assembleFluxTubeTensor_multiNfp_finiteBeta_randomAspect_nz96_51200tubes.pkl')
# tube names
tube_names_1 = tube_data_1['tube_files']
tube_names_2 = tube_data_2['tube_files']
tube_names_3 = tube_data_3['tube_files']
tube_names_4 = tube_data_4['tube_files']

# load AE data
file_path = "AE_processed_data/"
AE_1 = np.load(file_path+'20240601-01-assembleFluxTubeMatrix_noShiftScale_nz96_withCvdrift_AE.npy').flatten()
AE_2 = np.load(file_path+'20240726-01-assembleFluxTubeTensor_vacuum_nz96_AE.npy').flatten()
AE_3 = np.load(file_path+'20241004-01-assembleFluxTubeTensor_multiNfp_finiteBeta_nz96_AE.npy').flatten()
AE_4 = np.load(file_path+'20241103-01-assembleFluxTubeTensor_multiNfp_finiteBeta_randomAspect_nz96_51200tubes_AE.npy').flatten()
# get the dictionary
Q_1 = data_1['Q_avgs_without_FSA_grad_x'].flatten()
Q_2 = data_2['Q_avgs_without_FSA_grad_x'].flatten()
Q_3 = data_3['Q_avgs_without_FSA_grad_x'].flatten()
Q_4 = data_4['Q_avgs_without_FSA_grad_x'].flatten()

# concatenate the data
AE_all = np.concatenate((AE_1,AE_2,AE_3,AE_4)).flatten()
Q_all = np.concatenate((Q_1,Q_2,Q_3,Q_4)).flatten()
tube_names_all = np.concatenate((tube_names_1,tube_names_2,tube_names_3,tube_names_4)).flatten()

# split the data by Nfp, by searching for the tube names. name contains "nfpx"
idx_nfp2 = [i for i, name in enumerate(tube_names_all) if 'nfp2' in name]
idx_nfp3 = [i for i, name in enumerate(tube_names_all) if 'nfp3' in name]
idx_nfp4 = [i for i, name in enumerate(tube_names_all) if 'nfp4' in name]
idx_nfp5 = [i for i, name in enumerate(tube_names_all) if 'nfp5' in name]

# scatter plot, split by Nfp. Q1 and Q2 are nfp4, Q3 is nfp2,3,4,5, Q4 is nfp2,3,4,5
# labels
nfp4_AE = AE_all[idx_nfp4]
nfp4_Q = Q_all[idx_nfp4]
nfp3_AE = AE_all[idx_nfp3]
nfp3_Q = Q_all[idx_nfp3]
nfp2_AE = AE_all[idx_nfp2]
nfp2_Q = Q_all[idx_nfp2]
nfp5_AE = AE_all[idx_nfp5]
nfp5_Q = Q_all[idx_nfp5]

# plot the scatter. Many points (>10^5) so use alpha to make it more visible
fig, ax = plt.subplot_mosaic([['nfp2','.','nfp3'],
                                ['.','all','.'],
                                ['nfp4','.','nfp5']],constrained_layout=True,figsize=(12,8))
s_val=1
alpha_val=0.05
marker_shape='o'

# plot nfp2 on top left, nfp3 on top right, nfp4 on center left, nfp5 on center right, all on bottom
ax['nfp2'].scatter(nfp2_AE,nfp2_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:green')
ax['nfp3'].scatter(nfp3_AE,nfp3_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:orange')
ax['nfp4'].scatter(nfp4_AE,nfp4_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:blue')
ax['nfp5'].scatter(nfp5_AE,nfp5_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:red')
ax['all'].scatter(nfp2_AE,nfp2_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:green')
ax['all'].scatter(nfp3_AE,nfp3_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:orange')
ax['all'].scatter(nfp4_AE,nfp4_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:blue')
ax['all'].scatter(nfp5_AE,nfp5_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:red')

# set labels, scales, and limits
for key in ax.keys():
    ax[key].set_xlabel(r'$\widehat{A}$')
    ax[key].set_ylabel(r'$\widehat{Q}$')
    ax[key].set_xscale('log')
    ax[key].set_yscale('log')
    ax[key].set_xlim([1e-3,1e0])
    ax[key].set_ylim([1e-2,1e3])

# add titles
ax['nfp2'].set_title(r'$N_{fp}=2$')
ax['nfp3'].set_title(r'$N_{fp}=3$')
ax['nfp4'].set_title(r'$N_{fp}=4$')
ax['nfp5'].set_title(r'$N_{fp}=5$')

# fit a 3/2 power law to the data
fit_func = lambda logx, loga: loga + 3/2*logx
# fit the data
from scipy.optimize import curve_fit
popt, pcov = curve_fit(fit_func,np.log(AE_all),np.log(Q_all))
print('loga: ',popt[0])
# plot the fit
x_fit = np.linspace(1e-3,1e0,100)
y_fit = np.exp(fit_func(np.log(x_fit),*popt))
# add to all plots
for key in ax.keys():
    ax[key].plot(x_fit,y_fit,color='black',linestyle='--',label=r'$Q \propto A^{3/2}$')
    ax[key].legend(loc='upper right',fontsize=8)

# save the figure
plt.savefig('scatter_AE_Q.png',dpi=1000)

plt.show()