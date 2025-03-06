import IO
import numpy as np
import scipy.integrate as spi
import multiprocessing as mp

path = IO.ralfs_path

# load the data
file_name   = path+"output_file_1.h5"
data        = IO.load_data(file_name)

# get Qs, and gradients from the data
Qs = []
fprims = []
tprims = []
tube_name = []
for tube in data.keys():
    tube_data = data[tube]
    input = tube_data['input']
    output = tube_data['output']
    # get the Q
    Q = output['Q_ave_norm']
    # get fprim & tprim attributes
    fprim = input.attrs['fprim'][0]
    tprim = input.attrs['tprim'][0]
    # append to the list
    Qs.append(Q)
    fprims.append(fprim)
    tprims.append(tprim)
# convert to numpy arrays
Qs = np.array(Qs)
fprims = np.array(fprims)
tprims = np.array(tprims)

# loop over tubes in parallel
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    print('Number of processors: ',mp.cpu_count())
    print('Number of tubes: ',len(data.keys()))
    # use calc_AE from IO.py
    results = [pool.starmap_async(IO.calc_AE, [(data,idx_tube,Qs[idx_tube],fprims[idx_tube],tprims[idx_tube])]) for idx_tube in range(len(data.keys()))]
    results_list = [r.get() for r in results]
    # save data as hdf5
    file_path = IO.AE_path
    # retrieve the 2024... part of the file name
    file_save = file_name.split('-')[-3]+'_RANDOM.hdf5'
    # get rid of the /
    file_save = file_save.split('/')[-1]
    print('Saving to: ',file_path+file_save)
    IO.save_to_hdf5(results_list,file_path,file_save)