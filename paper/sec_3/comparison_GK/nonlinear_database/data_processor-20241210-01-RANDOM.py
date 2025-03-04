from IO import *
import numpy as np
import scipy.integrate as spi
import multiprocessing as mp

path = IO.matts_path

# load the data
file_name   = path+"/20241210-01-assembleFluxTubeTensor_allConfigs_filtered.pkl"
file_Q      = path+"/20241210-01-GX_results_for_random_gradients_allConfigs_filtered.pkl"
data        = load_data(file_name)
data_Q      = load_data(file_Q)





# check number of tubes
n_tubes = data['n_tubes']
# initialize arrays
AEs = np.zeros(n_tubes)
Qs = data_Q['Q_avgs_without_FSA_grad_x'][0:n_tubes]
fprims = data_Q['fprims'][0:n_tubes]
tprims = data_Q['tprims'][0:n_tubes]
# loop over tubes in parallel
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    print('Number of processors: ',mp.cpu_count())
    print('Number of tubes: ',n_tubes)
    # use calc_AE from IO.py
    results = [pool.starmap_async(calc_AE, [(data,idx_tube,Qs[idx_tube],fprims[idx_tube],tprims[idx_tube])]) for idx_tube in range(n_tubes)]
    results_list = [r.get() for r in results]
    # save data as hdf5
    file_path = IO.AE_path
    # retrieve the 2024... part of the file name
    file_save = file_name.split('-')[-3]+'.hdf5'
    # get rid of the /
    file_save = file_save.split('/')[-1]
    print('Saving to: ',file_path+file_save)
    save_to_hdf5(results_list,file_path,file_save)