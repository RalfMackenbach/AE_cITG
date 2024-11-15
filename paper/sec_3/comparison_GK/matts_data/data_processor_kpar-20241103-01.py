from IO import *
import numpy as np
import scipy.integrate as spi
import multiprocessing as mp
import source.ae_ITG as ae


def calc_kpar(data,idx_tube):
    dict = AE_dictionary(data,idx_tube)
    dict = function_extender(dict)
    # get relevant quantities
    B               = dict['bmag']
    gbdrift         = dict['gbdrift']
    l               = np.linspace(-1.0,1.0,len(B))
    # take the heaviside function of gbdrift
    gbdrift = np.heaviside(gbdrift,0)
    # take fourier transform of gbdrift and get the largest mode
    gbdrift_ft = np.fft.fft(gbdrift)
    kpar_idx = np.argmax(np.abs(gbdrift))
    kpar = 2*np.pi*kpar_idx/(2*len(B))
    print('Tube: ',idx_tube,' kpar: ',kpar)
    return kpar


# load the data
file_name   = "20241103-01-assembleFluxTubeTensor_multiNfp_finiteBeta_randomAspect_nz96_51200tubes.pkl"

data        = load_data(file_name)




# check number of tubes
n_tubes = data['n_tubes']
# initialize arrays
kpars = np.zeros(n_tubes)

# calculate kpar for each tube
for idx_tube in range(n_tubes):
    kpars[idx_tube] = calc_kpar(data,idx_tube)

# save the data as numpy array
file_path = "connection_length_estimate/"
np.save(file_path+file_name.split('.')[0]+'_kpar.npy',kpars)
