# from "scatter_data_GX_start0_end107.npz" and "scatter_data_GX_start8_end107.npz",
# check how much of the data is below 20% error threshold

from itertools import count
import re
from unicodedata import unidata_version
import numpy as np

# load the data from the npz files
data_fixed = np.load('scatter_data_GX_start8_end107.npz')
data_random = np.load('scatter_data_GX_start0_end107.npz')

# get the keys (labels) from the data
keys_fixed = list(data_fixed.keys())
keys_random = list(data_random.keys())

if True:
    # compare other key values against the first key, fixed data. Only on unstable data (where Q of the first key > 0.1)
    unstable_threshold = 0.1
    convergence_threshold = 1.2
    # make matrix to store the results of fixed (keys x # data points)
    data_number_fixed = len(data_fixed[keys_fixed[0]])
    results_fixed = np.zeros((len(keys_fixed), data_number_fixed))
    # fill the results matrix with the data
    for i, key in enumerate(keys_fixed):
        results_fixed[i, :] = data_fixed[key]
    unstable_counter = 0
    unstable_unconverged_counter = 0
    stable_counter = 0
    stable_unconverged_counter = 0
    # loop over each column in the results matrix
    for i in range(data_number_fixed):
        # get the column data
        column_data = results_fixed[:, i]
        # get the first value of the column
        first_value = column_data[0]
        # check if stable
        if first_value < unstable_threshold:
            stable_counter += 1
            # check if any other value is greater than the convergence threshold
            if np.any(column_data > unstable_threshold):
                stable_unconverged_counter += 1
        # check if unstable
        elif first_value > unstable_threshold:
            unstable_counter += 1
            # check if any other value is greater than the convergence threshold
            if np.any(column_data/first_value > convergence_threshold):
                unstable_unconverged_counter += 1
    # print the results
    print(f"Fixed stable data points: {stable_counter}, of which {stable_unconverged_counter} are unconverged, i.e. {stable_unconverged_counter/stable_counter*100:.2f}%")
    print(f"Fixed unstable data points: {unstable_counter}, of which {unstable_unconverged_counter} are unconverged, i.e. {unstable_unconverged_counter/unstable_counter*100:.2f}%")
        
# same thing for random data
if True:
    # compare other key values against the first key, random data. Only on unstable data (where Q of the first key > 0.1)
    unstable_threshold = 0.1
    convergence_threshold = 1.2
    # make matrix to store the results of random (keys x # data points)
    data_number_random = len(data_random[keys_random[0]])
    results_random = np.zeros((len(keys_random), data_number_random))
    # fill the results matrix with the data
    for i, key in enumerate(keys_random):
        results_random[i, :] = data_random[key]
    unstable_counter = 0
    unstable_unconverged_counter = 0
    stable_counter = 0
    stable_unconverged_counter = 0
    # loop over each column in the results matrix
    for i in range(data_number_random):
        # get the column data
        column_data = results_random[:, i]
        # get the first value of the column
        first_value = column_data[0]
        # check if stable
        if first_value < unstable_threshold:
            stable_counter += 1
            # check if any other value is greater than the convergence threshold
            if np.any(column_data > unstable_threshold):
                stable_unconverged_counter += 1
        # check if unstable
        elif first_value > unstable_threshold:
            unstable_counter += 1
            # check if any other value is greater than the convergence threshold
            if np.any(column_data/first_value > convergence_threshold):
                unstable_unconverged_counter += 1
    # print the results
    print(f"Random stable data points: {stable_counter}, of which {stable_unconverged_counter} are unconverged, i.e. {stable_unconverged_counter/stable_counter*100:.2f}%")
    print(f"Random unstable data points: {unstable_counter}, of which {unstable_unconverged_counter} are unconverged, i.e. {unstable_unconverged_counter/unstable_counter*100:.2f}%")