"""
This script automates the generation and submission of a large batch of gyrokinetic simulation jobs for a parameter scan on an HPC cluster using SLURM.
Includes varying gradients.

Main features:
- Defines a scan range and simulation parameters, including options for convergence studies.
- Cleans up old SLURM scripts and creates a new directory structure for the scan.
- For each simulation index, generates a random, valid input file using gx_gen_input_file.py utilities.
- Optionally generates additional input files for convergence studies by varying numerical parameters.
- Creates SLURM job scripts for each simulation (and convergence case if enabled) using a template.
- If enabled, submits all jobs to the cluster using sbatch, with throttling to avoid scheduler overload.
"""

import os
from gx_gen_input_file import *
import subprocess
import time
import numpy as np

# scan parameters
start_idx = 0                           # first index of convergence study. Needs to be a multiple of 4
end_idx   = 107                         # last index of convergence study.
size      = int(end_idx-start_idx+1)    # total number of sims
submit = False                          # auto-submit to slurm
convergence = True                      # add convergence folder and runs to the study

def gen_prims(idx):
    # set random seed
    if idx is not None:
        np.random.seed(idx)
    eta_angle_std   = np.arctan2(3.0, 0.9)
    dpdx_std        = 3.9
    angle_min       = np.arctan2(1.0, 3.0)
    # generate random values centered at std values
    while True:
        eta_angle = np.random.normal(eta_angle_std, 0.25)
        dpdx_rand = np.random.normal(dpdx_std, 1.0)
        angle_rand = np.arctan2(np.sin(eta_angle), np.cos(eta_angle))
        if dpdx_rand >= 0.5 and angle_min <= angle_rand <= (np.pi / 2 + 0.1):
                break
    # generate random int [0,1,2]
    p0, p1, p2 = 1/3, 1/3, 1/3  # probabilities for each case
    rand_int = np.random.choice([0, 1, 2], p=[p0, p1, p2])
    # if zero change both dpdx and eta
    if rand_int == 0:
        dpdx = dpdx_rand
        eta_angle = angle_rand
    # if one change only dpdx
    elif rand_int == 1:
        dpdx = dpdx_rand
        eta_angle = eta_angle_std
    # if two change only eta
    elif rand_int == 2:
        dpdx = dpdx_std
        eta_angle = angle_rand
    # calculate tprim and fprim
    tprim_unit = np.sin(eta_angle)
    fprim_unit = np.cos(eta_angle)
    # (tprim_unit + fprim_unit)*C = dpdx -> C = dpdx / (tprim_unit + fprim_unit)
    C = dpdx / (tprim_unit + fprim_unit)
    tprim = tprim_unit * C
    fprim = fprim_unit * C
    # if within delta of std fprim and tprim, resample
    delta = 0.1
    radius = np.sqrt((tprim - 3.0) ** 2 + (fprim - 0.9) ** 2)
    if radius < delta:
        return gen_prims(None)
    return float(tprim), float(fprim)


# first remove all files in slurm folder
arr = os.listdir('slurm/')
for val in arr:
    os.remove('slurm/'+val)



# string list for all convergence variables. Only used is convergence=True
# of the form [dbl/hlf]_[kwarg_key].in
string_list = ['dbl_nhermite.in', 'dbl_nlaguerre.in', 'dbl_ntheta.in', 'dbl_nx.in', 
               'dbl_ny.in', 'dbl_t_max.in', 'dbl_y0.in', 'hlf_D_hyper.in', 
               'dbl_jmult.in', 'hlf_cfl.in']  #,'hlf_vnewk.in'] # exclude hlf_vnewk.in in statistical study

# path to write files
scratch_path = os.environ['SCRATCH']


folder_name='GX_start{}_end{}'.format(start_idx,end_idx)
folder_path = scratch_path+'/'+folder_name
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)
else:
    raise ValueError('Folder already exists!')

# place logs in log_dir
log_dir = folder_path + '/' + 'log_dir'
os.mkdir(log_dir)

# make folders with names GX_start to GX_end
for i in range(start_idx,end_idx+1):
    
    # generate random prims
    tprim, fprim = gen_prims(i)

    # make simulation folder
    simulations_folder = folder_path+'/'+'GX_{}'.format(i)
    os.mkdir(simulations_folder)

    # populate folder with input file(s)
    # get the base-case
    std_dict, std_kwargs = return_input_dict(i,tprim,fprim)

    if convergence:
        dicts = [None]*len(string_list)
        # now vary numerical parameters
        for idx, val in enumerate(string_list):
            # check if dbl of hlf
            if val[0:3]=='dbl':
                multiplier = 2
            if val[0:3]=='hlf':
                multiplier = 1/2
            # get the kwarg key
            key = val[4:-3]
            # special clause for nperiod (also double ntheta), y0, etc.
            if key == 'nperiod':
                args = {'scan_idx': i, 'tprim': tprim, 'fprim': fprim, key: std_kwargs[key]*multiplier, 'ntheta': std_kwargs['ntheta']*multiplier}
            elif key == 'y0':
                args = {'scan_idx': i, 'tprim': tprim, 'fprim': fprim, key: std_kwargs[key]*multiplier, 'ny': std_kwargs['ny']*multiplier, 'nx': std_kwargs['nx']*multiplier}
            elif key == 'jmult':
                args = {'scan_idx': i, 'tprim': tprim, 'fprim': fprim, key: std_kwargs[key]*multiplier, 'nx': std_kwargs['nx']*multiplier}
            else:
                args = {'scan_idx': i, 'tprim': tprim, 'fprim': fprim, key: std_kwargs[key]*multiplier}

            dicts[idx],_=return_input_dict(**args)

    # generate input files
    # standard in main directory
    file_creator(std_dict,simulations_folder+'/input.in')

    if convergence:
        # convergence checks in convergence folder
        convergence_folder = simulations_folder + '/' + 'convergence_study'
        os.mkdir(convergence_folder)
        for idx, _ in enumerate(dicts):
            path = convergence_folder+'/'+string_list[idx]
            file_creator(dicts[idx],path)


    # now generate submit scripts, one node for each simulation
    # Read in the file
    with open('slurm_template', 'r') as file:
        filedata = file.read()
        conv_filedata = filedata
        # Replace the target string
        filedata = filedata.replace('INPUT_NAME', str(i))
        filedata = filedata.replace('OUTPUT_PATH', log_dir)
        filedata = filedata.replace('ERROR_PATH', log_dir)
        filedata = filedata.replace('SIM', folder_path+'/'+'GX_{}'.format(i)+'/'+'input.in')

        if convergence:
            # make copies for each string
            file_list   = [conv_filedata]*len(string_list)
            for idx, string in enumerate(string_list):
                key = string[4:-3]
                # Replace the target strings
                file_list[idx] = file_list[idx].replace('INPUT_NAME', str(i) + key)
                file_list[idx] = file_list[idx].replace('OUTPUT_PATH', log_dir)
                file_list[idx] = file_list[idx].replace('ERROR_PATH', log_dir)
                file_list[idx] = file_list[idx].replace('SIM', folder_path+'/'+'GX_{}'.format(i)+'/'+'convergence_study/'+string_list[idx])

    # Write the file out again
    with open('slurm/slurm_{}'.format(i), 'w') as file:
        file.write(filedata)

    # if convergence, same for convergence files
    if convergence:
        for idx, string in enumerate(string_list):
            conv_file = file_list[idx]
            key = string[:-3]
            with open('slurm/slurm_'+key+'_{}'.format(i), 'w') as file:
                file.write(conv_file)

    # # now generate submit scripts
    # if i % 4 == 3:
    #     # Read in the file
    #     with open('slurm_template', 'r') as file:
    #         filedata = file.read()
    #         conv_filedata = filedata
    #         # Replace the target string
    #         filedata = filedata.replace('INPUT_NAME', str(i))
    #         filedata = filedata.replace('OUTPUT_PATH', log_dir)
    #         filedata = filedata.replace('ERROR_PATH', log_dir)
    #         filedata = filedata.replace('SIM_0', folder_path+'/'+'GX_{}'.format(i-3)+'/'+'input.in')
    #         filedata = filedata.replace('SIM_1', folder_path+'/'+'GX_{}'.format(i-2)+'/'+'input.in')
    #         filedata = filedata.replace('SIM_2', folder_path+'/'+'GX_{}'.format(i-1)+'/'+'input.in')
    #         filedata = filedata.replace('SIM_3', folder_path+'/'+'GX_{}'.format(i-0)+'/'+'input.in')

    #         if convergence:
    #             # make copies for each string
    #             file_list   = [conv_filedata]*len(string_list)
    #             for idx, string in enumerate(string_list):
    #                 key = string[4:-3]
    #                 # Replace the target strings
    #                 file_list[idx] = file_list[idx].replace('INPUT_NAME', str(i) + key)
    #                 file_list[idx] = file_list[idx].replace('OUTPUT_PATH', log_dir)
    #                 file_list[idx] = file_list[idx].replace('ERROR_PATH', log_dir)
    #                 file_list[idx] = file_list[idx].replace('SIM_0', folder_path+'/'+'GX_{}'.format(i-3)+'/'+'convergence_study/'+string_list[idx])
    #                 file_list[idx] = file_list[idx].replace('SIM_1', folder_path+'/'+'GX_{}'.format(i-2)+'/'+'convergence_study/'+string_list[idx])
    #                 file_list[idx] = file_list[idx].replace('SIM_2', folder_path+'/'+'GX_{}'.format(i-1)+'/'+'convergence_study/'+string_list[idx])
    #                 file_list[idx] = file_list[idx].replace('SIM_3', folder_path+'/'+'GX_{}'.format(i-0)+'/'+'convergence_study/'+string_list[idx])

    #     # Write the file out again
    #     with open('slurm/slurm_{}'.format(i), 'w') as file:
    #         file.write(filedata)

    #     # if convergence, same for convergence files
    #     if convergence:
    #         for idx, string in enumerate(string_list):
    #             conv_file = file_list[idx]
    #             key = string[:-3]
    #             with open('slurm/slurm_'+key+'_{}'.format(i), 'w') as file:
    #                 file.write(conv_file)


if submit:
    # if submit, check all submit files in submit folder, and sbatch
    arr = os.listdir('slurm/')

    bash_command = "cd /users/rmackenb/gx_scan/slurm"
    subprocess.run([bash_command],capture_output=True,shell=True)


    job_idx = 0
    for val in arr:
        job_idx += 1
        sbatch_command = 'sbatch /users/rmackenb/gx_scan/slurm/{}'.format(val)
        proc = subprocess.run([sbatch_command],capture_output=True,shell=True)
        print(proc.stdout)
        time.sleep(0.01)
        # every n jobs, sleep for m seconds
        if job_idx % 500 == 0:
            time.sleep(20)