from __future__ import absolute_import, division, print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
import imageio
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import numpy as np
import h5py
from netCDF4 import Dataset

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data", "FilesToOctreeify")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")



if __name__ == '__main__':
    f = h5py.File(os.path.join(data_folder, 'Plume.h5'), 'r')
    d = torch.tensor(f.get('data'))
    d = d[:,128:256,63,:].unsqueeze(0)
    print(d.shape)
    for i in range(1, 3):
        rootgrp = Dataset(os.path.join(project_folder_path, "cut_down"+str(i)+".nc"), "w", format="NETCDF4")
        rootgrp.createDimension("u")
        rootgrp.createDimension("v")

        dim_0 = rootgrp.createVariable("data", np.float32, ("u","v"))
        dim_0[:] = d[0,0,::int(2**i), ::int(2**i)].cpu().numpy()

    f.close()

    f = h5py.File(os.path.join(data_folder, 'PlumeCut.h5'), 'w')
    f.create_dataset('data', data=d[0])

