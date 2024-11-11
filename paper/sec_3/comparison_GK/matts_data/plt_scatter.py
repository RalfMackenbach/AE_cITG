# plot the scatter
import matplotlib.pyplot as plt
from IO import *

# enable latex rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# load Q data
data_1 = load_data('20240601-01-103_gx_nfp4_production_gx_results_gradxRemoved.pkl')
data_2 = load_data('20240726-01-random_stellarator_equilibria_and_GX_gx_results_gradxRemoved.pkl')
data_3 = load_data('20241004-01-random_stellarator_equilibria_GX_results_combined.pkl')

# load AE data
file_path = "AE_processed_data_normal_lengthscales_NOT_functionextended/"
AE_1 = np.load(file_path+'20240601-01-assembleFluxTubeMatrix_noShiftScale_nz96_withCvdrift_AE.npy')
AE_2 = np.load(file_path+'20240726-01-assembleFluxTubeTensor_vacuum_nz96_AE.npy')
AE_3 = np.load(file_path+'20241004-01-assembleFluxTubeTensor_multiNfp_finiteBeta_nz96_AE.npy')

# get the dictionary
Q_1 = data_1['Q_avgs_without_FSA_grad_x']
Q_2 = data_2['Q_avgs_without_FSA_grad_x']
Q_3 = data_3['Q_avgs_without_FSA_grad_x']

# labels 
label_1 = r'$N_{fp}=4, \; \beta>0$'
label_2 = r'$N_{fp}=4, \; \beta=0$'
label_3 = r'$N_{fp}=\{2,3,4,5\}, \; \beta>0$'

# plot the scatter. Many points (>10^5) so use alpha to make it more visible
fig, ax = plt.subplots(1,2,constrained_layout=True,figsize=(8,4))
s_val=1
alpha_val=0.05
marker_shape='o'
ax[0].scatter(AE_3,Q_3,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:blue')
ax[0].scatter(AE_2,Q_2,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:orange')
ax[0].scatter(AE_1,Q_1,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:green')
ax[0].set_xlabel(r'$\widehat{A}$')
ax[0].set_ylabel(r'$\widehat{Q}$')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
# same thing for second plot, but with limits
ax[1].scatter(AE_3,Q_3,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:blue')
ax[1].scatter(AE_2,Q_2,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:orange')
ax[1].scatter(AE_1,Q_1,alpha=alpha_val,s=s_val,marker=marker_shape,color='tab:green')
ax[1].set_xlabel(r'$\widehat{A}$')
ax[1].set_ylabel(r'$\widehat{Q}$')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlim([2e-3,5e-1])
ax[1].set_ylim([1e-2,1e3])

# add legend manually with alpha=1
ax[1].scatter([],[],label=label_1,alpha=1,s=1,marker=marker_shape,color='tab:green')
ax[1].scatter([],[],label=label_2,alpha=1,s=1,marker=marker_shape,color='tab:orange')
ax[1].scatter([],[],label=label_3,alpha=1,s=1,marker=marker_shape,color='tab:blue')
ax[1].legend(loc='upper right',fontsize=8)

# save the figure
plt.savefig('scatter_AE_Q.png',dpi=1000)

plt.show()




# plt.figure()
# s_val=5
# alpha_val=0.05
# marker_shape='o'
# plt.scatter(AE_3,Q_3,alpha=alpha_val,label='20241004-01',s=s_val,marker=marker_shape)
# plt.scatter(AE_2,Q_2,alpha=alpha_val,label='20240726-01',s=s_val,marker=marker_shape)
# plt.scatter(AE_1,Q_1,alpha=alpha_val,label='20240601-01',s=s_val,marker=marker_shape)
# plt.xlabel(r'$\widehat{A}$')
# plt.ylabel(r'$\widehat{Q}$')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()