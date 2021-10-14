import torch
from torch import tensor
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
import matplotlib.pyplot as plt


def get_location2D(full_height: int, full_width : int, depth : int, index : int) -> Tuple[int, int]:
    final_x : int = 0
    final_y : int = 0

    current_depth : int = depth
    current_index : int = index
    while(current_depth > 0):
        s_x = int(full_width / (2**current_depth))
        s_y = int(full_height / (2**current_depth))
        x_offset = s_x * int((current_index % 4) / 2)
        y_offset = s_y * (current_index % 2)
        final_x += x_offset
        final_y += y_offset
        current_depth -= 1
        current_index = int(current_index / 4)

    return (final_x, final_y)

def get_location3D(full_width : int, full_height: int, full_depth : int, 
depth : int, index : int) -> Tuple[int, int, int]:
    final_x : int = 0
    final_y : int = 0
    final_z : int = 0

    current_depth : int = depth
    current_index : int = index
    while(current_depth > 0):
        s_x = int(full_width / (2**current_depth))
        s_y = int(full_height / (2**current_depth))
        s_z = int(full_depth / (2**current_depth))

        x_offset = s_x * int((current_index % 8) / 4)
        y_offset = s_y * int((current_index % 4) / 2)
        z_offset = s_z * (current_index % 2)
        
        final_x += x_offset
        final_y += y_offset
        final_z += z_offset
        current_depth -= 1
        current_index = int(current_index / 8)

    return (final_x, final_y, final_z)

def add_node_to_data_caches(node: OctreeNode, full_shape: List[int],
data_levels: List[torch.Tensor], mask_levels: List[torch.Tensor]):
    curr_ds_ratio = (2**node.LOD)
    if(len(node.data.shape) == 4):
        x_start, y_start = get_location2D(full_shape[2], full_shape[3], node.depth, node.index)
        ind = node.LOD 
        data_levels[ind][:,:,
            int(x_start/curr_ds_ratio): \
            int(x_start/curr_ds_ratio)+node.data.shape[2],
            int(y_start/curr_ds_ratio): \
            int(y_start/curr_ds_ratio)+node.data.shape[3]
        ] = node.data
        mask_levels[ind][:,:,
            int(x_start/curr_ds_ratio): \
            int(x_start/curr_ds_ratio)+node.data.shape[2],
            int(y_start/curr_ds_ratio): \
            int(y_start/curr_ds_ratio)+node.data.shape[3],
        ] = 1
    elif(len(node.data.shape) == 5):
        x_start, y_start, z_start = get_location3D(full_shape[2], full_shape[3], full_shape[4], node.depth, node.index)
        ind = node.LOD
        
        data_levels[ind][:,:,
            int(x_start/curr_ds_ratio): \
            int(x_start/curr_ds_ratio)+node.data.shape[2],
            int(y_start/curr_ds_ratio): \
            int(y_start/curr_ds_ratio)+node.data.shape[3],
            int(z_start/curr_ds_ratio): \
            int(z_start/curr_ds_ratio)+node.data.shape[4]
        ] = node.data
        mask_levels[ind][:,:,
            int(x_start/curr_ds_ratio): \
            int(x_start/curr_ds_ratio)+node.data.shape[2],
            int(y_start/curr_ds_ratio): \
            int(y_start/curr_ds_ratio)+node.data.shape[3],
            int(z_start/curr_ds_ratio): \
            int(z_start/curr_ds_ratio)+node.data.shape[4]
        ] = 1

def create_caches_from_octree(octree: OctreeNodeList, 
    full_shape : List[int]) -> \
    Tuple[List[torch.Tensor], List[torch.Tensor], 
    List[torch.Tensor], List[torch.Tensor]]:

    data_levels: List[torch.Tensor] = []
    mask_levels: List[torch.Tensor] = []
    data_downscaled_levels: List[torch.Tensor] = []
    mask_downscaled_levels: List[torch.Tensor] = []
    curr_LOD = 0

    device = octree[0].data.device
    max_LOD = octree.max_LOD()

    if(len(octree[0].data.shape) == 4):
        curr_shape = [full_shape[0], full_shape[1], full_shape[2], full_shape[3]]
    else:
        curr_shape = [full_shape[0], full_shape[1], full_shape[2], full_shape[3], full_shape[4]]

    while(curr_LOD <= max_LOD):
        data_levels.append(torch.zeros(curr_shape, dtype=torch.float32).to(device))
        data_downscaled_levels.append(torch.zeros(curr_shape, dtype=torch.float32).to(device))
        mask_levels.append(torch.zeros(curr_shape, dtype=torch.bool).to(device))
        mask_downscaled_levels.append(torch.zeros(curr_shape, dtype=torch.bool).to(device))
        curr_shape[2] = int(curr_shape[2] / 2)
        curr_shape[3] = int(curr_shape[3] / 2)  
        if(len(octree[0].data.shape) == 5):
            curr_shape[4] = int(curr_shape[4] / 2)
        curr_LOD += 1
        
    for i in range(len(octree)):
        add_node_to_data_caches(octree[i], full_shape,
            data_levels, mask_levels)
    
    return data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels

def octree_to_downscaled_levels(max_LOD : int,
    data_levels: List[torch.Tensor], mask_levels:List[torch.Tensor],
    data_downscaled_levels: List[torch.Tensor], mask_downscaled_levels:List[torch.Tensor],
    mode : str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    mask_downscaled_levels[0][:] = mask_levels[0][:]
    data_downscaled_levels[0][:] = data_levels[0][:]

    curr_LOD = 1
    while curr_LOD <= max_LOD:

        data_down = downscale(data_downscaled_levels[curr_LOD-1])
        if(mode == "2D"):
            mask_down = mask_downscaled_levels[curr_LOD-1][:,:,::2,::2]
        if(mode == "3D"):
            mask_down = mask_downscaled_levels[curr_LOD-1][:,:,::2,::2,::2]

        data_downscaled_levels[curr_LOD] = data_down + data_levels[curr_LOD]
        mask_downscaled_levels[curr_LOD] = mask_down + mask_levels[curr_LOD]

        curr_LOD += 1

    return data_downscaled_levels, mask_downscaled_levels

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

def generate_patch(z,z_stop,y,y_stop,x,x_stop,available_gpus):

    device = None
    while(device is None):        
        device, generator, input_volume = available_gpus.get_next_available()
        time.sleep(1)
    #print("Starting SR on device " + device)
    with torch.no_grad():
        result = generator(input_volume[:,:,z:z_stop,y:y_stop,x:x_stop])
    return result,z,z_stop,y,y_stop,x,x_stop,device

def generate_by_patch_parallel(generator, input_volume, patch_size, receptive_field, devices):
    with torch.no_grad():
        final_volume = torch.zeros(
            [input_volume.shape[0], input_volume.shape[1], input_volume.shape[2]*2, 
            input_volume.shape[3]*2, input_volume.shape[4]*2],
            dtype=torch.float32).to(devices[0])
        
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

class UpscalingMethod(nn.Module):
    def __init__(self, method : str, device : str, model_name = None,
        distributed = False):
        super(UpscalingMethod, self).__init__()
        self.method : str = method
        self.device : str = device
        self.models = []
        self.distributed = distributed
        self.devices = []
        if(self.method == "model"):            
            self.load_models(model_name, device, distributed)
        
    def load_models(self, model_name, device, distributed):
        with torch.no_grad():
            project_folder_path = os.path.dirname(os.path.abspath(__file__))
            project_folder_path = os.path.join(project_folder_path, "..")
            models_folder = os.path.join(project_folder_path, "SavedModels")
            opt = load_options(os.path.join(models_folder, model_name))
            opt["device"] = device

            generators, _ = load_models(opt,"cpu")
            for i in range(len(generators)):
                generators[i] = generators[i].to(opt['device'])
                generators[i].train(False)
            self.models = generators
        torch.cuda.empty_cache()
        if(distributed and torch.cuda.device_count() > 1):
            for i in range(torch.cuda.device_count()):
                self.devices.append("cuda:"+str(i))
        print("Loaded models")

    def forward(self, in_frame : torch.Tensor, scale_factor : float,
    lod : Optional[int] = None) -> torch.Tensor:
        with torch.no_grad():
            if(self.method == "linear"):
                up = F.interpolate(in_frame, 
                    mode='bilinear' if len(in_frame.shape) == 4 else "trilinear", 
                    scale_factor=scale_factor)
            elif(self.method == "bicubic"):
                up = F.interpolate(in_frame, mode='bicubic', scale_factor=scale_factor)
            elif(self.method == "nearest"):
                up = F.interpolate(in_frame, mode="nearest", scale_factor=scale_factor)
            elif(self.method == "model"):
                up = in_frame
                while(scale_factor > 1):
                    if(len(self.models) - lod < 0):
                        print("Model not supported for downscaling level " + str(lod) + \
                            ", using interpolation instead")
                        # Use interpolation instead
                        up = F.interpolate(up, 
                            mode='bilinear' if len(in_frame.shape) == 4 else "trilinear", 
                            scale_factor=2)
                    else:
                        if not self.distributed:
                            up = self.models[len(self.models)-lod](up)
                        else:
                            up = generate_by_patch_parallel(self.models[len(self.models)-lod], 
                                up, 140, 10, self.devices)
                    scale_factor = int(scale_factor / 2)
                    lod -= 1
            else:
                print("No support for upscaling method: " + str(self.method))
        return up

def upscale_volume(octree, full_shape, upscale):
    data_levels, mask_levels, data_downscaled_levels, mask_downscaled_levels = \
        create_caches_from_octree(octree, full_shape)

    data_downscaled_levels, mask_downscaled_levels = \
        octree_to_downscaled_levels(octree.max_LOD(),
        data_levels, mask_levels, data_downscaled_levels, 
        mask_downscaled_levels, "2D" if len(octree[0].data.shape) == 4 else "3D")

    curr_LOD = octree.max_LOD()

    restored_volume = data_downscaled_levels[curr_LOD].clone()
    step = 0
    tensor_to_nc(restored_volume, str(step)+'_orig.nc')
    step += 1
    while(curr_LOD > 0):
        
        restored_volume = upscale(restored_volume, 2, curr_LOD)
        tensor_to_nc(restored_volume, str(step)+'_upscale.nc')
        torch.cuda.synchronize()
        curr_LOD -= 1

        restored_volume *= (~mask_downscaled_levels[curr_LOD])
        data_downscaled_levels[curr_LOD] *= mask_downscaled_levels[curr_LOD] 
        restored_volume += data_downscaled_levels[curr_LOD]
        tensor_to_nc(restored_volume, str(step)+'_replace.nc')
        step += 1

    while(len(data_downscaled_levels) > 0):
        del data_downscaled_levels[0]
        del mask_downscaled_levels[0]
        
    step += 1
    return restored_volume

def upscale_volume_seams(octree: OctreeNodeList, full_shape: List[int], 
    upscale):
    restored_volume = torch.zeros(full_shape).to(octree[0].data.device)
    
    for i in range(len(octree)):
        curr_node = octree[i]
        tensor_to_nc(curr_node.data, "node_"+str(i)+".nc")
        if(len(octree[0].data.shape) == 4):
            x_start, y_start = get_location2D(full_shape[2], full_shape[3], curr_node.depth, curr_node.index)
            img_part = upscale(curr_node.data, 2**curr_node.LOD, curr_node.LOD)
            restored_volume[:,:,x_start:x_start+img_part.shape[2],y_start:y_start+img_part.shape[3]] = img_part
        else:
            x_start, y_start, z_start = get_location3D(full_shape[2], full_shape[3], full_shape[4], curr_node.depth, curr_node.index)
            img_part = upscale(curr_node.data, 2**curr_node.LOD, curr_node.LOD)
            restored_volume[:,:,x_start:x_start+img_part.shape[2],y_start:y_start+img_part.shape[3],z_start:z_start+img_part.shape[4]] = img_part
        tensor_to_nc(img_part, "node_"+str(i)+"_upscaled.nc")
    return restored_volume

def upscale_volume_downscalinglevels(octree: OctreeNodeList, full_shape: List[int], 
    border=False) \
    -> Tuple[torch.Tensor, torch.Tensor]:
    device = octree[0].data.device

    if(len(octree[0].data.shape) == 4):
        full_img = torch.zeros([full_shape[0], 3, full_shape[2], full_shape[3]]).to(device)
    elif(len(octree[0].data.shape) == 5):
        full_img = torch.zeros([full_shape[0], 3, full_shape[2], full_shape[3], full_shape[4]]).to(device)
    # palette here: https://www.pinterest.com/pin/432978951683169251/?d=t&mt=login
    # morandi colors
    # https://colorbrewer2.org/#type=sequential&scheme=YlOrRd&n=6
    cmap : List[torch.Tensor] = [
        torch.tensor([[255,255,255]], dtype=octree[0].data.dtype, device=device),
        torch.tensor([[255,255,178]], dtype=octree[0].data.dtype, device=device),
        torch.tensor([[254,217,118]], dtype=octree[0].data.dtype, device=device),
        torch.tensor([[254,178,76]], dtype=octree[0].data.dtype, device=device),
        torch.tensor([[253,141,60]], dtype=octree[0].data.dtype, device=device),
        torch.tensor([[252,78,42]], dtype=octree[0].data.dtype, device=device),
        torch.tensor([[227,26,28]], dtype=octree[0].data.dtype, device=device),        
        torch.tensor([[177,0,38]], dtype=octree[0].data.dtype, device=device)
    ]
    for i in range(len(cmap)):
        cmap[i] = cmap[i].unsqueeze(2).unsqueeze(3)
        if(len(octree[0].data.shape) == 5):
            cmap[i] = cmap[i].unsqueeze(4)

    for i in range(len(octree)):
        curr_node = octree[i]
        if(len(octree[0].data.shape) == 4):
            x_start, y_start = get_location2D(full_shape[2], full_shape[3], curr_node.depth, curr_node.index)
            s : int = curr_node.LOD
            if(border):
                full_img[:,:,
                    int(x_start): \
                    int(x_start)+ \
                        int((curr_node.data.shape[2]*(2**curr_node.LOD))),
                    int(y_start): \
                    int(y_start)+ \
                        int((curr_node.data.shape[3]*(2**curr_node.LOD)))
                ] = torch.zeros([full_shape[0], 3, 
                curr_node.data.shape[2]*(2**curr_node.LOD),
                curr_node.data.shape[3]*(2**curr_node.LOD)])
                full_img[:,:,
                    int(x_start)+1: \
                    int(x_start)+ \
                        int((curr_node.data.shape[2]*(2**curr_node.LOD)))-1,
                    int(y_start)+1: \
                    int(y_start)+ \
                        int((curr_node.data.shape[3]*(2**curr_node.LOD)))-1
                ] = cmap[s].repeat(full_shape[0], 1, 
                int((curr_node.data.shape[2]*(2**curr_node.LOD)))-2, 
                int((curr_node.data.shape[3]*(2**curr_node.LOD)))-2)
            else:
                full_img[:,:,
                    int(x_start): \
                    int(x_start)+ \
                        int((curr_node.data.shape[2]*(2**curr_node.LOD))),
                    int(y_start): \
                    int(y_start)+ \
                        int((curr_node.data.shape[3]*(2**curr_node.LOD)))
                ] = cmap[s].repeat(full_shape[0], 1, 
                int((curr_node.data.shape[2]*(2**curr_node.LOD))), 
                int((curr_node.data.shape[3]*(2**curr_node.LOD))))
        elif(len(octree[0].data.shape) == 5):
            x_start, y_start, z_start = get_location3D(full_shape[2], full_shape[3], full_shape[4],
            curr_node.depth, curr_node.index)
            s : int = curr_node.LOD
            if(border):
                full_img[:,:,
                    int(x_start): \
                    int(x_start)+ \
                        int((curr_node.data.shape[2]*(2**curr_node.LOD))),
                    int(y_start): \
                    int(y_start)+ \
                        int((curr_node.data.shape[3]*(2**curr_node.LOD))),
                    int(z_start): \
                    int(z_start)+ \
                        int((curr_node.data.shape[4]*(2**curr_node.LOD)))
                ] = torch.zeros([full_shape[0], 3, 
                curr_node.data.shape[2]*(2**curr_node.LOD),
                curr_node.data.shape[3]*(2**curr_node.LOD),
                curr_node.data.shape[4]*(2**curr_node.LOD)])
                full_img[:,:,
                    int(x_start)+1: \
                    int(x_start)+ \
                        int((curr_node.data.shape[2]*(2**curr_node.LOD)))-1,
                    int(y_start)+1: \
                    int(y_start)+ \
                        int((curr_node.data.shape[3]*(2**curr_node.LOD)))-1,
                    int(z_start)+1: \
                    int(z_start)+ \
                        int((curr_node.data.shape[4]*(2**curr_node.LOD)))-1
                ] = cmap[s].repeat(full_shape[0], 1, 
                int((curr_node.data.shape[2]*(2**curr_node.LOD)))-2, 
                int((curr_node.data.shape[3]*(2**curr_node.LOD)))-2,
                int((curr_node.data.shape[4]*(2**curr_node.LOD)))-2)
            else:
               full_img[:,:,
                    int(x_start): \
                    int(x_start)+ \
                        int((curr_node.data.shape[2]*(2**curr_node.LOD))),
                    int(y_start): \
                    int(y_start)+ \
                        int((curr_node.data.shape[3]*(2**curr_node.LOD))),
                    int(z_start): \
                    int(z_start)+ \
                        int((curr_node.data.shape[4]*(2**curr_node.LOD)))
                ] = cmap[s].repeat(full_shape[0], 1, 
                int((curr_node.data.shape[2]*(2**curr_node.LOD))), 
                int((curr_node.data.shape[3]*(2**curr_node.LOD))),
                int((curr_node.data.shape[4]*(2**curr_node.LOD)))) 
    cmap_img_height : int = 64
    cmap_img_width : int = 512
    cmap_img = torch.zeros([cmap_img_width, cmap_img_height, 3], dtype=torch.float, device=device)
    y_len : int = int(cmap_img_width / len(cmap))
    for i in range(len(cmap)):
        y_start : int = i * y_len
        y_end : int = (i+1) * y_len
        cmap_img[y_start:y_end, :, :] = torch.squeeze(cmap[i])

    return full_img, cmap_img

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    parser.add_argument('--octree_file',default="Plume.octree",type=str,help='Octree file to load and super resolve')
    parser.add_argument('--volume_file',default="Plume.h5",type=str,help='File to octree-ify. Should be of shape [c, h, w, [d]]')
    parser.add_argument('--save_name',default="Plume_64xreduction",type=str,help='Name to save the upscaled result as for rendering')
    parser.add_argument('--upscaling_method',default="model",type=str,help='How to upscale the data. Nearest, linear, or model')
    parser.add_argument('--model_name',default="Plume_ESRGAN",type=str,help='Model name to use, if using model')    
    parser.add_argument('--distributed',default="False",type=str2bool,help='Whether or not to upscale the volume in parallel on GPUs available')
    parser.add_argument('--device',default="cuda:0",type=str)
    parser.add_argument('--save_original_volume',default="True",type=str2bool,help='Write out the original volume as an NC for visualization too.')
    parser.add_argument('--save_error_volume',default="False",type=str2bool,help='Write out the error volume as an NC for visualization too.')
    parser.add_argument('--save_downscaling_levels',default="True",
        type=str2bool,help='Write out the octree downscaling levels as images for visualization too.')
    parser.add_argument('--seams',default="False",
        type=str2bool,help='Upscale blocks individually instead of using our MRSR algorithm')
    parser.add_argument('--compute_metrics',default="True",type=str2bool,help='Compute PSNR/SSIM')
    parser.add_argument('--border_on_octree',default="True",type=str2bool,help='Show border on octree vis')

    args = vars(parser.parse_args())
    
    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    save_folder = os.path.join(project_folder_path, "Output", "UpscaledOctreeData")
    file_path = os.path.join(data_folder, "FilesToOctreeify", args['volume_file'])
    octree_path = os.path.join(data_folder, "OctreeFiles", args['octree_file'])
            
    print("Loading original volume from " + file_path)   
    f = h5py.File(file_path, 'r')
    d = torch.tensor(f.get('data'))
    f.close()
    volume = d.unsqueeze(0).to(args['device'])
    # for figure in paper with blockwise example
    #volume = volume[:,:,0:128,:,64]

    print("Loading octree from " + octree_path)
    #octree = torch.load(octree_path)
    octree = OctreeNodeList.load(octree_path, args['device'])

    upscale = UpscalingMethod(args['upscaling_method'], args['device'], 
        args['model_name'] if args['upscaling_method'] == "model" else None, 
        args['distributed'] if args['upscaling_method'] == 'model' else None)
    
    print("Upscaling volume")
    start_time = time.time()
    if(args['seams']):
        MRSR_volume = upscale_volume_seams(octree, volume.shape, upscale)
    else:
        MRSR_volume = upscale_volume(octree, volume.shape, upscale)
    end_time = time.time()
    print("It took %0.02f seconds to upscale the volume with %s" % \
        (end_time - start_time, args['upscaling_method']))

    if(args['upscaling_method'] == "model"):
        total_inference_calls = 0
        if(args['seams']):
            for i in range(len(octree)):
                total_inference_calls += octree[i].LOD
        else:
            total_inference_calls = octree.max_LOD()
        print("Total number of inference calls was %i" % total_inference_calls)
    if(args['compute_metrics']):
        p = PSNR_torch(MRSR_volume, volume).item()
        if(len(volume.shape) == 4):
            s = ssim(MRSR_volume, volume).item()
        elif len(volume.shape) == 5 and args['distributed']:
            s = ssim3D_distributed(MRSR_volume, volume).item()
        else:
            s = ssim3D(MRSR_volume, volume).item()

        errs = (MRSR_volume - volume).flatten().cpu().numpy()
        np.save(os.path.join(save_folder, args['save_name']+"_errs.npy"), errs)

        print("Average abs error: %0.06f, median abs error: %0.06f" % \
            (np.abs(errs).mean(), np.median(np.abs(errs))))

        plt.hist(errs, bins=100, range=(errs.mean()-errs.std()*2, errs.mean()+errs.std()*2))
        plt.title("Error histogram")
        plt.xlabel("Error")
        plt.ylabel("Occurance (proportion)")
        ys, _ = plt.yticks()
        ys = np.array(ys, dtype=float)
        plt.yticks(ys, np.around(ys / len(errs), 4))
        plt.savefig(os.path.join(save_folder, args['save_name']+"_twosided_err_histogram.png"))
        plt.clf()

        plt.hist(np.abs(errs), bins=100, range=(0, np.abs(errs).mean()+np.abs(errs).std()*2))
        plt.title("Absolute error histogram")
        plt.xlabel("Error")
        plt.ylabel("Occurance (proportion)")
        ys, _ = plt.yticks()
        ys = np.array(ys, dtype=float)
        plt.yticks(ys, np.around(ys / len(errs), 4))
        plt.savefig(os.path.join(save_folder, args['save_name']+"_err_histogram.png"))
        plt.clf()

    print("Saving upscaled volume to " + os.path.join(save_folder, args['save_name']+".nc"))
    tensor_to_nc(MRSR_volume, os.path.join(save_folder, args['save_name']+".nc"))

    if(args['save_original_volume']):
        print("Saving upscaled volume to " + os.path.join(save_folder, args['volume_file']+".nc"))
        tensor_to_nc(volume, os.path.join(save_folder, args['volume_file']+".nc"))

    if(args['compute_metrics'] and args['save_error_volume']):
        print("Saving error volume to " + os.path.join(save_folder, args['volume_file']+"_err.nc"))
        tensor_to_nc(torch.abs(volume[0,0]-MRSR_volume[0,0]).cpu().numpy(),
            os.path.join(save_folder, args['save_name']+"_err.nc"))

    if(args['save_downscaling_levels']):
        print("Saving downscaling level images to " + os.path.join(save_folder, args['volume_file']+".nc"))
        downscaling_levels_img, cmap_img = \
            upscale_volume_downscalinglevels(octree, volume.shape, args['border_on_octree'])
        if(len(downscaling_levels_img.shape) == 5):
            downscaling_levels_img = downscaling_levels_img[..., int(downscaling_levels_img.shape[-1]/2)+1]
        imageio.imwrite(os.path.join(save_folder, args['octree_file'] + ".png"),
            downscaling_levels_img.cpu()[0].permute(1, 2, 0).numpy())
        imageio.imwrite(os.path.join(save_folder, args['octree_file'] + "_cmap.png"),
            cmap_img.cpu().numpy())

    print()
    if(args['compute_metrics']):
        print("################################# Statistics/metrics #################################")
        print()
        print("PSNR: %0.02f, SSIM: %0.02f" % (p, s))
