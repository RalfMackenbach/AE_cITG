import numpy as np
import matplotlib.pyplot as plt
# enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# enable \boldsymbol (requires a LaTeX package)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

plot=True

# load the data from the npz files
data_fixed = np.load('scatter_data_GX_start8_end107.npz')
data_random = np.load('scatter_data_GX_start0_end107.npz')


# get the keys (labels) from the data
keys_fixed = list(data_fixed.keys())
data_number_fixed = len(data_fixed[keys_fixed[0]])
results_fixed = np.zeros((len(keys_fixed), data_number_fixed))
# fill the results matrix with the data
for i, key in enumerate(keys_fixed):
    results_fixed[i, :] = data_fixed[key]
keys_random = list(data_random.keys())
data_number_random = len(data_random[keys_random[0]])
results_random = np.zeros((len(keys_random), data_number_random))
# fill the results matrix with the data
for i, key in enumerate(keys_random):
    results_random[i, :] = data_random[key]

max_err_fixed = np.zeros((data_number_fixed))
max_err_random = np.zeros((data_number_random))

# loop over each column in the results matrix (fixed data)
for i in range(data_number_fixed):
    # get the reference value (first value of the column)
    reference_fixed = results_fixed[0, i]
    # if reference is <0.1 set to nan
    if reference_fixed < 0.1:
        max_err_fixed[i] = np.nan
    else:
        # calculate the maximum absolute error for each key
        vals = np.abs(np.log2(results_fixed[:, i] / reference_fixed))
        max_err_fixed[i] = np.nanmax(vals)
# loop over each column in the results matrix (random data)
for i in range(data_number_random):
    # get the reference value (first value of the column)
    reference_random = results_random[0, i]
    # if reference is <0.1 set to nan
    if reference_random < 0.1:
        max_err_random[i] = np.nan
    else:
        # calculate the maximum absolute error for each key
        vals = np.abs(np.log2(results_random[:, i] / reference_random))
        max_err_random[i] = np.nanmax(vals)

# make histograms of the maximum absolute error
if plot:
    bins = np.linspace(0, 3, 30)
    # make one extra bin for all values between 3 and infinity
    last_bin = np.linspace(3, 1000000, 2)
    bins = np.concatenate((bins, last_bin))
    # make the histogram
    plt.hist(max_err_fixed, bins=bins, alpha=0.5, label='Fixed')
    plt.hist(max_err_random, bins=bins, alpha=0.5, label='Random')
    # add a vertical line at 20% 
    plt.axvline(x=np.log2(1.2), color='k', linestyle='--', label=r'$\log_2(1.2)$')
    # add legend
    plt.legend()
    plt.grid()
    plt.xlim(0, 4.0)
    plt.ylabel('Counts')
    plt.xlabel(r'$\max \left| \log_2 \boldsymbol{Q}_{\rm res} - \log_2 Q_{\rm nom} \right|$')
    # adjust last tick label to '>3'
    plt.gca().set_xticks(np.arange(0, 4, 0.5))
    plt.gca().set_xticklabels(np.concatenate((np.arange(0, 3.5, 0.5), [r'$>3$'])))
    # save and close the plot
    plt.savefig('absolute_error_histogram.png', dpi=1000)
    plt.close()

    # print how much of the data is below the 20% error threshold
    # remove NaN values from the max_err_fixed and max_err_random arrays
    non_nan_fixed = ~np.isnan(max_err_fixed)
    non_nan_random = ~np.isnan(max_err_random)
    below_threshold_fixed = np.sum(max_err_fixed[non_nan_fixed] < np.log2(1.2))
    below_threshold_random = np.sum(max_err_random[non_nan_random] < np.log2(1.2))
    total_fixed = np.sum(non_nan_fixed)
    total_random = np.sum(non_nan_random)
    print(f"Fixed data: {below_threshold_fixed}/{total_fixed} ({below_threshold_fixed/total_fixed*100:.2f}%) below log2(1.2) error threshold")
    print(f"Random data: {below_threshold_random}/{total_random} ({below_threshold_random/total_random*100:.2f}%) below log2(1.2) error threshold")
    # print how much of the data is below a factor 2 
    # remove NaN values from the max_err_fixed and max_err_random arrays
    below_factor_2_fixed = np.sum(max_err_fixed[non_nan_fixed] < np.log2(2))
    below_factor_2_random = np.sum(max_err_random[non_nan_random] < 1)
    print(f"Fixed data: {below_factor_2_fixed}/{total_fixed} ({below_factor_2_fixed/total_fixed*100:.2f}%) below factor 2 error threshold")
    print(f"Random data: {below_factor_2_random}/{total_random} ({below_factor_2_random/total_random*100:.2f}%) below factor 2 error threshold")