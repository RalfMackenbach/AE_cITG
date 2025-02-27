from IO import *
import numpy as np
import scipy.integrate as spi
import multiprocessing as mp
import source.ae_ITG as ae
import pandas as pd
import h5py


# load the data
file_name   = "ralfs_data/20250217-output_file.h5"
# file_name   = "matts_data/20240601-01-assembleFluxTubeMatrix_noShiftScale_nz96_withCvdrift.pkl"

# load hdf5 
data = h5py.File(file_name, 'r')
# load pickle
# data = load_data(file_name)

idx = np.random.randint(len(data.keys()))

dict_sim = AE_dictionary_hdf5(data,idx)