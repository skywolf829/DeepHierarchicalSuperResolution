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
import matplotlib.pyplot as plt

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")



def create_err_hist():
    uni = np.load(os.path.join(project_folder_path, "Output", 
        "UpscaledOctreeData", "Plume_uniform_ESRGAN_4x_errs.npy"))    
    hei = np.load(os.path.join(project_folder_path, "Output", 
        "UpscaledOctreeData", "Plume_psnrcomparison_errs.npy")) 
    low = min(uni.mean()-uni.std()*2, hei.mean()-hei.std()*2)
    hi = min(uni.mean()+uni.std()*2, hei.mean()+hei.std()*2)

    plt.hist(uni, bins=100, range=(low, hi), label="Uniform SR", alpha=0.75, color='tab:orange')
    plt.hist(hei, bins=100, range=(low, hi), label="Multiresolution SR", alpha=0.75, color='tab:blue')
    plt.title("Error histogram")
    plt.xlabel("Error")
    plt.ylabel("Occurance (proportion)")
    plt.legend()
    ys, _ = plt.yticks()
    ys = np.array(ys, dtype=float)
    plt.yticks(ys, np.around(ys / len(uni), 4))
    plt.show()
    plt.clf()

    
    hi = min(np.abs(uni).mean()+np.abs(uni).std()*2, np.abs(hei).mean()+np.abs(hei).std()*2)

    plt.hist(np.abs(uni), bins=100, range=(0, hi), label="Uniform SR", alpha=0.75, color='tab:orange')
    plt.hist(np.abs(hei), bins=100, range=(0, hi), label="Multiresolution SR", alpha=0.75, color='tab:blue')
    plt.title("Absolute error histogram")
    plt.xlabel("Error")
    plt.ylabel("Occurance (proportion)")
    plt.legend()
    ys, _ = plt.yticks()
    ys = np.array(ys, dtype=float)
    plt.yticks(ys, np.around(ys / len(uni), 4))
    plt.show()

if __name__ == '__main__':
    fold = os.path.join(data_folder, "SuperResolutionData", "Supernova")

    for file in os.listdir(os.path.join(fold, "TrainingData")):
        f = h5py.File(os.path.join(os.path.join(fold, "TrainingData"), file), 'r')
        d = torch.tensor(f.get('data')).unsqueeze(0)
        f.close()
        d = F.interpolate(d, size=[448, 448, 448])[0]

        f2 = h5py.File(os.path.join(fold, file), 'w')
        f2.create_dataset("data", data=d.cpu().numpy())
        f2.close()

    for file in os.listdir(os.path.join(fold, "TestingData")):
        f = h5py.File(os.path.join(os.path.join(fold, "TestingData"), file), 'r')
        d = torch.tensor(f.get('data')).unsqueeze(0)
        f.close()
        d = F.interpolate(d, size=[448, 448, 448])[0]
        
        f2 = h5py.File(os.path.join(fold, file), 'w')
        f2.create_dataset("data", data=d.cpu().numpy())
        f2.close()
