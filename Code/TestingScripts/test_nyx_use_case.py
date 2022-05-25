import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from models import load_models
from options import  load_options
from utility_functions import  str2bool, AvgPool3D, AvgPool2D
import os
import argparse
import time
from math import log2
from datasets import TestingDataset
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import copy
from utility_functions import ssim, ssim3D, ssim3D_distributed, save_obj, load_obj
import h5py
import numpy as np
from netCDF4 import Dataset

def mse_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    return ((GT-x)**2).mean()

def psnr_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    data_range = GT.max() - GT.min()
    return (20.0*torch.log10(data_range)-10.0*torch.log10(mse_func(GT, x, device)))

def mre_func(GT, x, device):
    GT = GT.to(device)
    x = x.to(device)
    data_range = GT.max() - GT.min()
    return (torch.abs(GT-x).max() / data_range)

def tensor_to_nc(volume, path):
    rootgrp = Dataset(path, "w", format="NETCDF4")
    rootgrp.createDimension("u")
    rootgrp.createDimension("v")
    if(len(volume.shape) == 5):
        rootgrp.createDimension("w")
    if(len(volume.shape) == 5):
        dim_0 = rootgrp.createVariable("data", np.float32, ("u","v","w"))
    else:
        dim_0 = rootgrp.createVariable("data", np.float32, ("u","v"))
    dim_0[:] = volume[0,0].cpu().numpy()

def generate_by_patch(generator, input_volume, patch_size, receptive_field, device):
    with torch.no_grad():
        final_volume = torch.zeros(
            [input_volume.shape[0], input_volume.shape[1], input_volume.shape[2]*2, 
            input_volume.shape[3]*2, input_volume.shape[4]*2]
            ).to(device)
        
        rf = receptive_field
                    
        z_done = False
        z = 0
        z_stop = min(input_volume.shape[2], z + patch_size)
        while(not z_done):
            if(z_stop == input_volume.shape[2]):
                z_done = True
            y_done = False
            y = 0
            y_stop = min(input_volume.shape[3], y + patch_size)
            while(not y_done):
                if(y_stop == input_volume.shape[3]):
                    y_done = True
                x_done = False
                x = 0
                x_stop = min(input_volume.shape[4], x + patch_size)
                while(not x_done):                        
                    if(x_stop == input_volume.shape[4]):
                        x_done = True
                    #print("%d:%d, %d:%d, %d:%d" % (z, z_stop, y, y_stop, x, x_stop))
                    result = generator(input_volume[:,:,z:z_stop,y:y_stop,x:x_stop])

                    x_offset = rf if x > 0 else 0
                    y_offset = rf if y > 0 else 0
                    z_offset = rf if z > 0 else 0

                    final_volume[:,:,
                    2*z+z_offset:2*z+result.shape[2],
                    2*y+y_offset:2*y+result.shape[3],
                    2*x+x_offset:2*x+result.shape[4]] = result[:,:,z_offset:,y_offset:,x_offset:]

                    x += patch_size - 2*rf
                    x = min(x, max(0, input_volume.shape[4] - patch_size))
                    x_stop = min(input_volume.shape[4], x + patch_size)
                y += patch_size - 2*rf
                y = min(y, max(0, input_volume.shape[3] - patch_size))
                y_stop = min(input_volume.shape[3], y + patch_size)
            z += patch_size - 2*rf
            z = min(z, max(0, input_volume.shape[2] - patch_size))
            z_stop = min(input_volume.shape[2], z + patch_size)

    return final_volume

def generate_patch(z,z_stop,y,y_stop,x,x_stop,available_gpus):

    device = None
    while(device is None):        
        device, generator, input_volume = available_gpus.get_next_available()
        time.sleep(1)
    #print("Starting SR on device " + device)
    with torch.no_grad():
        result = generator(input_volume[:,:,z:z_stop,y:y_stop,x:x_stop])
    return result,z,z_stop,y,y_stop,x,x_stop,device

class SharedList(object):  
    def __init__(self, items, generators, input_volumes):
        self.lock = threading.Lock()
        self.list = items
        self.generators = generators
        self.input_volumes = input_volumes
        
    def get_next_available(self):
        #print("Waiting for a lock")
        self.lock.acquire()
        item = None
        generator = None
        input_volume = None
        try:
            #print('Acquired a lock, counter value: ', self.counter)
            if(len(self.list) > 0):                    
                item = self.list.pop(0)
                generator = self.generators[item]
                input_volume = self.input_volumes[item]
        finally:
            #print('Released a lock, counter value: ', self.counter)
            self.lock.release()
        return item, generator, input_volume
    
    def add(self, item):
        #print("Waiting for a lock")
        self.lock.acquire()
        try:
            #print('Acquired a lock, counter value: ', self.counter)
            self.list.append(item)
        finally:
            #print('Released a lock, counter value: ', self.counter)
            self.lock.release()

def generate_by_patch_parallel(generator, input_volume, patch_size, receptive_field, devices):
    with torch.no_grad():
        final_volume = torch.zeros(
            [input_volume.shape[0], input_volume.shape[1], input_volume.shape[2]*2, 
            input_volume.shape[3]*2, input_volume.shape[4]*2]
            ).to(devices[0])
        
        rf = receptive_field

        available_gpus = []
        generators = {}
        input_volumes = {}

        for i in range(1, len(devices)):
            available_gpus.append(devices[i])
            g = copy.deepcopy(generator).to(devices[i])
            iv = input_volume.clone().to(devices[i])
            generators[devices[i]] = g
            input_volumes[devices[i]] = iv
            torch.cuda.empty_cache()

        available_gpus = SharedList(available_gpus, generators, input_volumes)

        threads= []
        with ThreadPoolExecutor(max_workers=len(devices)-1) as executor:
            z_done = False
            z = 0
            z_stop = min(input_volume.shape[2], z + patch_size)
            while(not z_done):
                if(z_stop == input_volume.shape[2]):
                    z_done = True
                y_done = False
                y = 0
                y_stop = min(input_volume.shape[3], y + patch_size)
                while(not y_done):
                    if(y_stop == input_volume.shape[3]):
                        y_done = True
                    x_done = False
                    x = 0
                    x_stop = min(input_volume.shape[4], x + patch_size)
                    while(not x_done):                        
                        if(x_stop == input_volume.shape[4]):
                            x_done = True
                        
                        
                        threads.append(
                            executor.submit(
                                generate_patch,
                                z,z_stop,
                                y,y_stop,
                                x,x_stop,
                                available_gpus
                            )
                        )
                        
                        x += patch_size - 2*rf
                        x = min(x, max(0, input_volume.shape[4] - patch_size))
                        x_stop = min(input_volume.shape[4], x + patch_size)
                    y += patch_size - 2*rf
                    y = min(y, max(0, input_volume.shape[3] - patch_size))
                    y_stop = min(input_volume.shape[3], y + patch_size)
                z += patch_size - 2*rf
                z = min(z, max(0, input_volume.shape[2] - patch_size))
                z_stop = min(input_volume.shape[2], z + patch_size)

            for task in as_completed(threads):
                result,z,z_stop,y,y_stop,x,x_stop,device = task.result()
                result = result.to(devices[0])
                x_offset_start = rf if x > 0 else 0
                y_offset_start = rf if y > 0 else 0
                z_offset_start = rf if z > 0 else 0
                x_offset_end = rf if x_stop < input_volume.shape[4] else 0
                y_offset_end = rf if y_stop < input_volume.shape[3] else 0
                z_offset_end = rf if z_stop < input_volume.shape[2] else 0
                #print("%d, %d, %d" % (z, y, x))
                final_volume[:,:,
                2*z+z_offset_start:2*z+result.shape[2] - z_offset_end,
                2*y+y_offset_start:2*y+result.shape[3] - y_offset_end,
                2*x+x_offset_start:2*x+result.shape[4] - x_offset_end] = result[:,:,
                z_offset_start:result.shape[2]-z_offset_end,
                y_offset_start:result.shape[3]-y_offset_end,
                x_offset_start:result.shape[4]-x_offset_end]
                available_gpus.add(device)
    
    return final_volume

def get_test_results(GT, x, mode, distributed=False, test_on_gpu=True):
    if(not test_on_gpu):
        GT = GT.cpu()
        x = x.cpu()
    
    p = psnr_func(GT, x, GT.device).item()
    ms = mse_func(GT, x, GT.device).item()
    mr = mre_func(GT, x, GT.device).item()
    if(mode == "2D"):
        s = ssim(GT, x).item()
    else:
        if not distributed:
            s = ssim3D(GT, x).item()
        else:            
            s = ssim3D_distributed(GT, x).item()

    return {"PSNR (dB)": p, "SSIM": s, "MSE": ms, "MRE": mr}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    
    parser.add_argument('--mode',default="3D",type=str,help='2D or 3D')
    parser.add_argument('--model_name',default="nyx_use_case",type=str,help='The folder with the model to load')
    parser.add_argument('--device',default="cuda:0",type=str,help='Device to use for testing')
    
    args = vars(parser.parse_args())


    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..", "..")
    data_folder = os.path.join(project_folder_path, "Data", "SuperResolutionData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    print("Loading options and model")
    opt = load_options(os.path.join(save_folder, args["model_name"]))

    opt["device"] = args["device"]

    
    if(opt['model'] == "SSRTVD"):
        generators, _, _ = load_models(opt,"cpu")
    else:
        generators, _ = load_models(opt,"cpu")
    for i in range(len(generators)):
        generators[i] = generators[i].to(opt['device'])
        generators[i].train(False)

    lr_data_path = os.path.join(data_folder, "nyx64", "TestingDataLR")
    hr_data_path = os.path.join(data_folder, "nyx64", "TestingDataHR")

    results_location = os.path.join(output_folder, 'Nyx_use_case.results')
    
    if args['mode'] == "2D":
        interp = "bilinear"
    else:
        interp = "trilinear"

    saved_one = False

    with torch.no_grad():
        upscaling_results = {
            interp: {
                "Upscaling time": [],
                "MSE": [],
                "PSNR (dB)": [],
                "SSIM": [],
                "MRE": []
            },
            'model': {
                "Upscaling time": [],
                "MSE": [],
                "PSNR (dB)": [],
                "SSIM": [],
                "MRE": []
            }
        }
        files = os.listdir(lr_data_path)
        np.random.seed(2)
        np.random.shuffle(files)
        for f_name in files:
            print(f_name)
            f = h5py.File(os.path.join(lr_data_path, f_name))
            LR_data = torch.tensor(np.array(f['data'])).to(args['device']).unsqueeze(0)
            f.close()
            f = h5py.File(os.path.join(hr_data_path, f_name))
            GT_data = torch.tensor(np.array(f['data'])).to(args['device']).unsqueeze(0)
            f.close()
            print("LR Data size: " + str(LR_data.shape))
            print("HR Data size: " + str(GT_data.shape))
            
            #LR_data = AvgPool3D(GT_data.clone(), 4)
            inference_start_time = time.time()
            
            x = LR_data.clone()
            current_ds = 4
            while(current_ds > 1):
                gen_to_use = int(len(generators) - log2(current_ds))        
                if(args['mode'] == '3D'):
                    x = generate_by_patch(generators[gen_to_use], 
                        x, 64, 10, args['device'])
                elif(args['mode'] == '2D'):
                    x = generators[gen_to_use](x)                    
                current_ds = int(current_ds / 2)
                
            inference_end_time = time.time()                
            inference_this_frame = inference_end_time - inference_start_time

            print("Finished super resolving in %0.04f seconds. Start shape: %s, Final shape: %s. Performing tests." % \
                (inference_this_frame, str(LR_data.shape), str(x.shape)))
            frame_results = get_test_results(GT_data, 
                                                x, 
                                                "3D", 
                                                False,
                                                True)
            m = x.clone()
            
            print("Model: " + str(frame_results))
            upscaling_results['model']['Upscaling time'].append(inference_this_frame)
            for k in frame_results.keys():
                upscaling_results['model'][k].append(frame_results[k])

            inference_start_time = time.time()
            x = LR_data.clone()
            if(args['mode'] == "3D"):
                x = F.interpolate(x, scale_factor=4, 
                mode=interp, align_corners=False)
            elif(args['mode'] == '2D'):
                x = F.interpolate(x, scale_factor=4, 
                mode=interp, align_corners=False)
            inference_end_time = time.time()                
            inference_this_frame = inference_end_time - inference_start_time

            frame_results = get_test_results(GT_data, 
                                                x, 
                                                "3D", 
                                                False,
                                                True)
            print("Interpolation: " + str(frame_results))
            upscaling_results[interp]['Upscaling time'].append(inference_this_frame)
            for k in frame_results.keys():
                upscaling_results[interp][k].append(frame_results[k])

            if(not saved_one):
                tensor_to_nc(LR_data, os.path.join(output_folder, "LR.nc"))
                tensor_to_nc(x, os.path.join(output_folder, "Interp.nc"))
                tensor_to_nc(m, os.path.join(output_folder, "Model.nc"))
                tensor_to_nc(GT_data, os.path.join(output_folder, "GT.nc"))

                print(f"Saved {f_name}")
                saved_one=True

    save_obj(upscaling_results, results_location)
    print("Saved results")
