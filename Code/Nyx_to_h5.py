from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
from utility_functions import create_folder
from netCDF4 import Dataset
import h5py

def load_to_numpy(path, resolution):
    d = np.fromfile(path, dtype=np.float32)
    d = d.reshape((resolution, resolution, resolution))
    d = np.expand_dims(d, 0)
    return d

def np_to_NC(data, path):
    rootgrp = Dataset(path, "w", format="NETCDF4")
    rootgrp.createDimension("x")
    rootgrp.createDimension("y")
    rootgrp.createDimension("z")
    dim_0 = rootgrp.createVariable("data", np.float32, ("x","y","z"))
    dim_0[:] = data[0]

def np_to_h5(data, path):
    d = h5py.File(path, mode='w')
    d.create_dataset("data", data=data)
    d.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Calculate and save FTLE for data')
    parser.add_argument('--load_folder_name',default=None,type=str,help='Input folder')
    parser.add_argument('--save_folder_name',default=None,type=str,help='Save folder name')
    parser.add_argument('--res',default=64,type=int,help='resolution')
    
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    load_folder = os.path.join(project_folder_path, "Data", 
                               "SuperResolutionData", args['load_folder_name'])
    save_folder = os.path.join(project_folder_path, "Data", 
                               "SuperResolutionData", args['save_folder_name'])
    create_folder(os.path.join(project_folder_path, "Data", 
                               "SuperResolutionData"), args['save_folder_name'])
    
    for file in os.listdir(load_folder):
        print(file)
        full_path = os.path.join(load_folder, file)
        data = load_to_numpy(full_path, args['res'])
        data = np.log(data+1)
        data -= data.min()
        data /= data.max()
        np_to_h5(data, os.path.join(save_folder, file.split(".bin")[0]+".h5"))
        
    print(data.shape)
    np_to_NC(data, os.path.join(save_folder, file.split(".bin")[0]+".nc"))
    np_to_h5(data, os.path.join(save_folder, file.split(".bin")[0]+".h5"))
