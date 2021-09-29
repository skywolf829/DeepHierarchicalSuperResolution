import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from utility_functions import str2bool, PSNR_torch, ssim, ssim3D, ssim3D_distributed
import time
import h5py
import argparse
import os
from create_octree import OctreeNodeList, OctreeNode, downscale
from netCDF4 import Dataset
import numpy as np
from typing import List, Tuple, Optional
from options import load_options
from models import load_models
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import threading
import imageio
from octree_upscaling import UpscalingMethod, SharedList, generate_patch, generate_by_patch_parallel
from math import log2


def upscale_volume(volume, factor, upscale):
    
    restored_volume = volume.clone()
    curr_LOD = int(log2(factor))
    while(curr_LOD > 0):        
        restored_volume = upscale(restored_volume, 2, curr_LOD)
        torch.cuda.synchronize()
        curr_LOD -= 1

    return restored_volume


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    parser.add_argument('--volume_file',default="Plume.h5",type=str,help='File to octree-ify. Should be of shape [c, h, w, [d]]')    
    parser.add_argument('--scale_factor',default=2,type=int,help='Scale factor to do SR, power of 2')
    parser.add_argument('--save_name',default="Plume_uniform_upscaled_4x.nc",type=str,help='Name to save the upscaled result as for rendering')
    parser.add_argument('--upscaling_method',default="model",type=str,help='How to upscale the data. Nearest, linear, or model')
    parser.add_argument('--model_name',default="Plume",type=str,help='Model name to use, if using model')    
    parser.add_argument('--distributed',default="False",type=str2bool,help='Whether or not to upscale the volume in parallel on GPUs available')
    parser.add_argument('--device',default="cuda:0",type=str)
    parser.add_argument('--save_original_volume',default="False",type=str2bool,help='Write out the original volume as an NC for visualization too.')
    args = vars(parser.parse_args())
    
    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    save_folder = os.path.join(project_folder_path, "Output", "UpscaledOctreeData")
    file_path = os.path.join(data_folder, "FilesToOctreeify", args['volume_file'])
            
    print("Loading original volume from " + file_path)   
    f = h5py.File(file_path, 'r')
    d = torch.tensor(f.get('data'))
    f.close()
    volume = d.unsqueeze(0).to(args['device'])

    upscale = UpscalingMethod(args['upscaling_method'], args['device'], 
        args['model_name'] if args['upscaling_method'] == "model" else None, 
        args['distributed'] if args['upscaling_method'] == 'model' else None)
    
    print("Upscaling volume")
    start_time = time.time()
    SR_volume = upscale_volume(volume, volume.shape, upscale)
    end_time = time.time()
    print("It took %0.02f seconds to upscale the volume with %s" % \
        (end_time - start_time, args['upscaling_method']))

    p = PSNR_torch(SR_volume, volume).item()
    if(len(volume.shape) == 4):
        s = ssim(SR_volume, volume).item()
    elif len(volume.shape) == 5 and args['distributed']:
        s = ssim3D_distributed(SR_volume, volume).item()
    else:
        s = ssim3D(SR_volume, volume).item()

    print("Saving upscaled volume to " + os.path.join(save_folder, args['save_name']+".nc"))
    rootgrp = Dataset(os.path.join(save_folder, args['save_name']+".nc"), "w", format="NETCDF4")
    rootgrp.createDimension("u")
    rootgrp.createDimension("v")
    if(len(SR_volume.shape) == 5):
        rootgrp.createDimension("w")

    if(len(SR_volume.shape) == 5):
        dim_0 = rootgrp.createVariable("data", np.float32, ("u","v","w"))
    else:
        dim_0 = rootgrp.createVariable("data", np.float32, ("u","v"))
    dim_0[:] = SR_volume[0,0].cpu().numpy()

    if(args['save_original_volume']):
        print("Saving upscaled volume to " + os.path.join(save_folder, args['volume_file']+".nc"))
        rootgrp = Dataset(os.path.join(save_folder, args['volume_file']+".nc"), "w", format="NETCDF4")
        rootgrp.createDimension("u")
        rootgrp.createDimension("v")
        if(len(volume.shape) == 5):
            rootgrp.createDimension("w")

        if(len(volume.shape) == 5):
            dim_0 = rootgrp.createVariable("data", np.float32, ("u","v","w"))
        else:
            dim_0 = rootgrp.createVariable("data", np.float32, ("u","v"))
        dim_0[:] = volume[0,0].cpu().numpy()

    print()
    print("################################# Statistics/metrics #################################")
    print()

    print("PSNR: %0.02f, SSIM: %0.02f" % (p, s))
    print("Upscaling time: %0.02f" % (end_time-start_time))
