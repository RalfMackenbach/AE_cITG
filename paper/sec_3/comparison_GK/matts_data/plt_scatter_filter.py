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

# load kpar data
kpar_1 = np.load('connection_length_estimate/20240601-01-assembleFluxTubeMatrix_noShiftScale_nz96_withCvdrift_kpar.npy')
kpar_2 = np.load('connection_length_estimate/20240726-01-assembleFluxTubeTensor_vacuum_nz96_kpar.npy')
kpar_3 = np.load('connection_length_estimate/20241004-01-assembleFluxTubeTensor_multiNfp_finiteBeta_nz96_kpar.npy')

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
nfp4_kpar = np.concatenate((kpar_1,kpar_2,kpar_3[idx_nfp4])).flatten()
nfp3_AE = (AE_3[idx_nfp3]).flatten()
nfp3_Q = (Q_3[idx_nfp3]).flatten()
nfp3_kpar = (kpar_3[idx_nfp3]).flatten()
nfp2_AE = (AE_3[idx_nfp2]).flatten()
nfp2_Q = (Q_3[idx_nfp2]).flatten()
nfp2_kpar = (kpar_3[idx_nfp2]).flatten()
nfp5_AE = (AE_3[idx_nfp5]).flatten()
nfp5_Q = (Q_3[idx_nfp5]).flatten()
nfp5_kpar = (kpar_3[idx_nfp5]).flatten()



# make scatter plot of AE, Q, filtering by kpar
fig, ax = plt.subplots(constrained_layout=True)

# make mask based on where kpar is small
kpar_filter = 0.0
mask_nfp2 = nfp2_kpar <= kpar_filter
mask_nfp3 = nfp3_kpar <= kpar_filter
mask_nfp4 = nfp4_kpar <= kpar_filter
mask_nfp5 = nfp5_kpar <= kpar_filter

alpha_val = 0.1
s_val = 1

ax.scatter(nfp2_AE*mask_nfp2,nfp2_Q*mask_nfp2,alpha=alpha_val,s=s_val,marker='o',color='tab:green')
ax.scatter(nfp3_AE*mask_nfp3,nfp3_Q*mask_nfp3,alpha=alpha_val,s=s_val,marker='o',color='tab:orange')
ax.scatter(nfp4_AE*mask_nfp4,nfp4_Q*mask_nfp4,alpha=alpha_val,s=s_val,marker='o',color='tab:blue')
ax.scatter(nfp5_AE*mask_nfp5,nfp5_Q*mask_nfp5,alpha=alpha_val,s=s_val,marker='o',color='tab:red')

# set labels, scales, and limits
ax.set_xlabel(r'$\widehat{A}$')
ax.set_ylabel(r'$\widehat{Q}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e-3,1e0])
ax.set_ylim([1e-2,1e3])

# add power law fit
fit_func = lambda logx: 5.843843818250206 + 3/2*logx
x_fit = np.linspace(1e-3,1e0,100)  
y_fit = np.exp(fit_func(np.log(x_fit)))
ax.plot(x_fit,y_fit,color='black',linestyle='--',label=r'$Q \propto A^{3/2}$')

# add labels, set alpha=1
ax.scatter([],[],label=r'$N_{fp}=2$',alpha=1,s=5,marker='o',color='tab:green')
ax.scatter([],[],label=r'$N_{fp}=3$',alpha=1,s=5,marker='o',color='tab:orange')
ax.scatter([],[],label=r'$N_{fp}=4$',alpha=1,s=5,marker='o',color='tab:blue')
ax.scatter([],[],label=r'$N_{fp}=5$',alpha=1,s=5,marker='o',color='tab:red')


# add legend
ax.legend(loc='upper right',fontsize=8)

# save the figure 
plt.savefig('scatter_AE_Q_filtered.png',dpi=1000)

# show
plt.show()

# # plot nfp2 on top left, nfp3 on top right, nfp4 on center left, nfp5 on center right, all on bottom
# ax['nfp2'].scatter(nfp2_AE,nfp2_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:green')
# ax['nfp3'].scatter(nfp3_AE,nfp3_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:orange')
# ax['nfp4'].scatter(nfp4_AE,nfp4_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:blue')
# ax['nfp5'].scatter(nfp5_AE,nfp5_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:red')
# ax['all'].scatter(nfp2_AE,nfp2_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:green')
# ax['all'].scatter(nfp3_AE,nfp3_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:orange')
# ax['all'].scatter(nfp4_AE,nfp4_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:blue')
# ax['all'].scatter(nfp5_AE,nfp5_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:red')


# # set labels, scales, and limits
# for key in ax.keys():
#     ax[key].set_xlabel(r'$\widehat{A}$')
#     ax[key].set_ylabel(r'$\widehat{Q}$')
#     ax[key].set_xscale('log')
#     ax[key].set_yscale('log')
#     ax[key].set_xlim([1e-3,1e0])
#     ax[key].set_ylim([1e-2,1e3])

# # add titles
# ax['nfp2'].set_title(r'$N_{fp}=2$')
# ax['nfp3'].set_title(r'$N_{fp}=3$')
# ax['nfp4'].set_title(r'$N_{fp}=4$')
# ax['nfp5'].set_title(r'$N_{fp}=5$')
# ax['all'].set_title(r'$N_{fp}=\{2,3,4,5\}$')

# # ax[0].scatter(nfp2_AE,nfp2_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:green')
# # ax[0].scatter(nfp3_AE,nfp3_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:orange')
# # ax[0].scatter(nfp4_AE,nfp4_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:blue')
# # ax[0].scatter(nfp5_AE,nfp5_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:red')
# # ax[0].set_xlabel(r'$\widehat{A}$')
# # ax[0].set_ylabel(r'$\widehat{Q}$')
# # ax[0].set_xscale('log')
# # ax[0].set_yscale('log')
# # # same thing for second plot, but with limits
# # ax[1].scatter(nfp4_AE,nfp4_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:blue')
# # ax[1].scatter(nfp3_AE,nfp3_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:orange')
# # ax[1].scatter(nfp2_AE,nfp2_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:green')
# # ax[1].scatter(nfp5_AE,nfp5_Q,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:red')
# # ax[1].set_xlabel(r'$\widehat{A}$')
# # ax[1].set_ylabel(r'$\widehat{Q}$')
# # ax[1].set_xscale('log')
# # ax[1].set_yscale('log')
# # ax[1].set_xlim([2e-3,5e-1])
# # ax[1].set_ylim([1e-3,1e3])

# # fit a 3/2 power law to the data
# AEs = np.asarray(np.concatenate((nfp4_AE,nfp3_AE,nfp2_AE,nfp5_AE))).flatten()
# Qs = np.asarray(np.concatenate((nfp4_Q,nfp3_Q,nfp2_Q,nfp5_Q))).flatten()
# fit_func = lambda logx, loga: loga + 3/2*logx
# # fit the data
# from scipy.optimize import curve_fit
# popt, pcov = curve_fit(fit_func,np.log(AEs),np.log(Qs))
# # plot the fit
# x_fit = np.linspace(1e-3,1e0,100)
# y_fit = np.exp(fit_func(np.log(x_fit),*popt))
# # add to all plots
# for key in ax.keys():
#     ax[key].plot(x_fit,y_fit,color='black',linestyle='--',label=r'$Q \propto A^{3/2}$')
#     ax[key].legend(loc='upper right',fontsize=8)


# # add legend manually with alpha=1
# # ax[1].scatter([],[],label=r'$N_{fp}=2$',alpha=1,s=5,marker=marker_shape,color='tab:green')
# # ax[1].scatter([],[],label=r'$N_{fp}=3$',alpha=1,s=5,marker=marker_shape,color='tab:orange')
# # ax[1].scatter([],[],label=r'$N_{fp}=4$',alpha=1,s=5,marker=marker_shape,color='tab:blue')
# # ax[1].scatter([],[],label=r'$N_{fp}=5$',alpha=1,s=5,marker=marker_shape,color='tab:red')
# # ax[1].legend(loc='upper right',fontsize=8)


# # save the figure
# plt.savefig('scatter_AE_Q.png',dpi=1000)

# plt.show()