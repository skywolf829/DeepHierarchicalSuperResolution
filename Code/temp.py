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
data_folder = os.path.join(project_folder_path, "Data", "FilesToOctreeify")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")



if __name__ == '__main__':
    uni = np.load(os.path.join(project_folder_path, "Plume_uniform_errs.npy"))    
    hei = np.load(os.path.join(project_folder_path, "Plume_hierarchical_errs.npy"))
    low = min(uni.mean()-uni.std()*2, hei.mean()-hei.std()*2)
    hi = min(uni.mean()+uni.std()*2, hei.mean()+hei.std()*2)

    plt.hist(uni, bins=100, range=(low, hi), label="Uniform SR", alpha=0.75)
    plt.hist(hei, bins=100, range=(low, hi), label="Multiresolution SR", alpha=0.75)
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

    plt.hist(np.abs(uni), bins=100, range=(0, hi), label="Uniform SR", alpha=0.75)
    plt.hist(np.abs(hei), bins=100, range=(0, hi), label="Multiresolution SR", alpha=0.75)
    plt.title("Absolute error histogram")
    plt.xlabel("Error")
    plt.ylabel("Occurance (proportion)")
    ys, _ = plt.yticks()
    ys = np.array(ys, dtype=float)
    plt.yticks(ys, np.around(ys / len(uni), 4))
    plt.show()