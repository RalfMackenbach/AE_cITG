from IO import *
import numpy as np
import scipy.integrate as spi
import multiprocessing as mp
import source.ae_ITG as ae
import pandas as pd

path = '/misc/ITG_databse/matts_data'

# load the data
file_name   = path+"/20241210-01-assembleFluxTubeTensor_allConfigs_filtered.pkl"
file_Q      = path+"/20241210-01-GX_results_for_fixed_gradients_allConfigs_filtered.pkl"
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
    # use calc_AE from IO.py
    results = [pool.starmap_async(calc_AE, [(data,idx_tube,Qs[idx_tube],0.9,3.0)]) for idx_tube in range(n_tubes)]
    AEs = [r.get() for r in results]

    # AEs is now list of dictionaries. We convert it to a pandas dataframe
    AEs_df = pd.DataFrame(AEs)
    # order is AE_arr, k_alpha_arr, k_psi_arr, w_alpha, w_psi, B, AE_val, tube_idx, w_n, w_T, Q
    # mixed arrays and scalars.
    # save data as hdf5
    file_path = "AE_processed_data/"
    # retrieve the 2024... part of the file name
    file_save = file_name.split('-')[0]
    # remove the 'matts_data/' part
    file_save = file_save.split('/')[1]
    AEs_df.to_hdf(file_path+file_save+'-FIXED-AE_data.h5', key=file_name, mode='w')