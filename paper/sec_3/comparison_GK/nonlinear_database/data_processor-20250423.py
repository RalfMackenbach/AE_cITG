import IO
import numpy as np
import scipy.integrate as spi
import multiprocessing as mp
import h5py

path = IO.ralfs_path

# load the data
file_name   = path+"/prod_3_random.hdf5"
# load 
data        = h5py.File(file_name, 'r', swmr=True)

# get Qs, and gradients from the data
Qs = []
fprims = []
tprims = []
tube_name = []
kx_facs = []

for tube in data.keys():
    tube_data = data[tube]
    input = tube_data['input']
    geom = input['geom']
    output = tube_data['output']
    # get the Q
    Q = output['<heat-flux(t)>']
    # calculate flux-surface averaged |grad x|
    grad_x_squared  = geom['gds22_over_shat_squared']
    grad_x          = np.sqrt(grad_x_squared)
    bmag            = geom['B']
    # append first value at end of grad_x and bmag
    grad_x = np.append(grad_x, grad_x[0])
    bmag = np.append(bmag, bmag[0])
    theta_grid = np.linspace(-np.pi,np.pi,len(grad_x))
    # do FSA
    FSA = np.trapezoid(grad_x/bmag,theta_grid)/np.trapezoid(1.0/bmag,theta_grid)
    # get fprim & tprim attributes
    fprim = input.attrs['fprim'][0]
    tprim = input.attrs['tprim'][0]
    kxfac = geom['kxfac'][()]
    # fprim = fprim/kxfac
    # tprim = tprim/kxfac
    # finally adjust Q
    Q = Q * FSA / kxfac
    # append to the list
    Qs.append(Q)
    fprims.append(fprim)
    tprims.append(tprim)
    kx_facs.append(kxfac)
# convert to numpy arrays
Qs = np.array(Qs)
fprims = np.array(fprims)
tprims = np.array(tprims)
kx_facs = np.array(kx_facs)
# print(Qs)
# print(kx_facs)

# if __name__ == '__main__':
#     # do one calculation to see if it works
#     idx_tube = 0
#     # use calc_AE from IO.py
#     results = IO.calc_AE(data,idx_tube,Qs[idx_tube],fprims[idx_tube],tprims[idx_tube])


# check number of tubes
n_tubes = len(Qs)
n_subsample = 1
indices = range(0, n_tubes, n_subsample)
# loop over the tubes
def process_tube(idx_tube):
    ans = IO.calc_AE(data, idx_tube, Qs[idx_tube], fprims[idx_tube], tprims[idx_tube], pol_ext=False)
    return [ans]

if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_tube, indices)
    # save data as hdf5
    file_path = IO.AE_path
    file_save = 'TOK_3_RANDOM.hdf5'
    # get rid of the /
    file_save = file_save.split('/')[-1]
    print('Saving to: ', file_path + file_save)
    IO.save_to_hdf5(results, file_path, file_save)