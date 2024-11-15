from IO import *
import numpy as np
import scipy.integrate as spi
import multiprocessing as mp
import source.ae_ITG as ae


def calc_AE(data,idx_tube,plot=True):
    dict = AE_dictionary(data,idx_tube)
    dict = function_extender(dict)
    # first, we construct the length-scales
    B               = dict['bmag']
    g11             = dict['gds22_over_shat_squared']
    g22             = dict['gds2']
    grad_drift_y    = dict['gbdrift'] 
    grad_drift_x    = dict['gbdrift0_over_shat']
    # construct linspace -1,1 for arc-length coordinate
    l       = np.linspace(-1.0,1.0,len(B))
    Dx_varying = 1.0
    Dy_varying = 1.0
    Dx      = spi.simpson(Dx_varying/B,x=l)/spi.simpson(1/B,x=l)
    Dy      = spi.simpson(Dy_varying/B,x=l)/spi.simpson(1/B,x=l)
    w_T     = 3.0*Dx*np.ones_like(B)
    w_n     = 0.9*Dx*np.ones_like(B)
    w_alpha =-grad_drift_y*Dx
    w_psi   = grad_drift_x*Dy
    # now calculate the available energy at each l
    AE_arr  = np.zeros_like(l)
    k_alpha_arr = np.zeros_like(l)
    k_psi_arr = np.zeros_like(l)
    for idx, _ in np.ndenumerate(AE_arr):
        idx = idx[0]
        AE_dict = ae.calculate_AE(w_alpha[idx],w_psi[idx],w_n[idx],w_T[idx])
        AE_arr[idx] = AE_dict['AE']
        k_alpha_arr[idx] = AE_dict['k_alpha']
        k_psi_arr[idx] = AE_dict['k_psi']

    # now do integral of AE_arr
    AE = spi.simpson(AE_arr/B,x=l)/spi.simpson(1/B,x=l)
    # add both to dictionary
    dict['AE_arr'] = AE_arr
    dict['k_alpha_arr'] = k_alpha_arr
    dict['k_psi_arr'] = k_psi_arr
    dict['AE_val'] = AE
    if plot:
        plot_dict(dict)
    print('Tube: ',idx_tube,' AE: ',AE)
    return AE


# load the data
file_name   = "20241103-01-assembleFluxTubeTensor_multiNfp_finiteBeta_randomAspect_nz96_51200tubes.pkl"
file_Q      = "20241103-01-random_stellarator_equilibria_finiteBeta_randomAspect_allNFP_GX_results_combined.pkl"
data        = load_data(file_name)
data_Q      = load_data(file_Q)



# check number of tubes
n_tubes = data['n_tubes']
# initialize arrays
AEs = np.zeros(n_tubes)
Qs = data_Q['Q_avgs_without_FSA_grad_x'][0:n_tubes]
# loop over tubes in parallel
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    print('Number of processors: ',mp.cpu_count())
    print('Number of tubes: ',n_tubes)
    results = [pool.starmap_async(calc_AE, [(data,idx_tube,False)]) for idx_tube in range(n_tubes)]
    AEs = [r.get() for r in results]

    # save the data as numpy array
    file_path = "AE_processed_data/"
    np.save(file_path+file_name.split('.')[0]+'_AE.npy',AEs)

    # AEs = np.asarray(AEs).flatten()
    # Qs = np.asarray(Qs).flatten()

    # # scatter
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(AEs,Qs)
    # plt.xscale('log')
    # plt.yscale('log')
    # # do a linear fit and print the slope
    # from scipy.stats import linregress
    # slope, intercept, r_value, p_value, std_err = linregress(np.log(AEs),np.log(Qs))
    # print('Slope: ',slope)
    
    # plt.xlabel('AE')
    # plt.ylabel('Q')
    # plt.show()