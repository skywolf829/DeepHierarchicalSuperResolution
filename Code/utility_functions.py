import os
import numpy as np
import torch
from torch.nn import functional as F
from matplotlib.pyplot import cm
import pickle
from math import exp
from typing import Optional
import argparse

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
 
def save_obj(obj,location):
    with open(location, 'wb') as f:
        pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)

def load_obj(location):
    with open(location, 'rb') as f:
        return pickle.load(f)

def MSE(x, GT):
    return ((x-GT)**2).mean()

def PSNR(x, GT, max_diff = None):
    if(max_diff is None):
        max_diff = GT.max() - GT.min()
    p = 20 * np.log10(max_diff) - 10*np.log10(MSE(x, GT))
    return p

def relative_error(x, GT, max_diff = None):
    if(max_diff is None):
        max_diff = GT.max() - GT.min()
    val = np.abs(GT-x).max() / max_diff
    return val

def pw_relative_error(x, GT):
    val = np.abs(np.abs(GT-x) / GT)
    return val.max()

def gaussian(window_size : int, sigma : float) -> torch.Tensor:
    gauss : torch.Tensor = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x \
        in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size : torch.Tensor, channel : int) -> torch.Tensor:
    _1D_window : torch.Tensor = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window : torch.Tensor = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window : torch.Tensor = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1 : torch.Tensor, img2 : torch.Tensor, window : torch.Tensor, 
window_size : torch.Tensor, channel : int, size_average : Optional[bool] = True):
    mu1 : torch.Tensor = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 : torch.Tensor = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq : torch.Tensor = mu1.pow(2)
    mu2_sq : torch.Tensor = mu2.pow(2)
    mu1_mu2 : torch.Tensor = mu1*mu2

    sigma1_sq : torch.Tensor = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq : torch.Tensor = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 : torch.Tensor = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 : float = 0.01**2
    C2 : float= 0.03**2

    ssim_map : torch.Tensor = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    ans : torch.Tensor = torch.Tensor([0])
    if size_average:
        ans = ssim_map.mean()
    else:
        ans = ssim_map.mean(1).mean(1).mean(1)
    return ans

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1.to("cuda:2"), window.to("cuda:2"), padding = window_size//2, groups = channel).to("cuda:2")
    mu2 = F.conv3d(img2.to("cuda:3"), window.to("cuda:3"), padding = window_size//2, groups = channel).to("cuda:3")

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1.to("cuda:4")*mu2.to("cuda:4")

    sigma1_sq = F.conv3d(img1.to("cuda:4")*img1.to("cuda:4"), window.to("cuda:4"), padding = window_size//2, groups = channel).to("cuda:2") - mu1_sq.to("cuda:2")
    sigma2_sq = F.conv3d(img2.to("cuda:5")*img2.to("cuda:5"), window.to("cuda:5"), padding = window_size//2, groups = channel).to("cuda:3") - mu2_sq.to("cuda:3")
    sigma12 = F.conv3d(img1.to("cuda:6")*img2.to("cuda:6"), window.to("cuda:6"), padding = window_size//2, groups = channel) - mu1_mu2.to("cuda:6")

    C1 = 0.01**2
    C2 = 0.03**2

    #ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    mu1_sq += mu2_sq.to("cuda:2")
    mu1_sq += C1

    sigma1_sq += sigma2_sq.to("cuda:2")
    sigma1_sq += C2

    mu1_sq *= sigma1_sq

    mu1_mu2 *= 2
    mu1_mu2 += C1

    sigma12 *= 2
    sigma12 += C2

    mu1_mu2 *= sigma12.to("cuda:4")

    mu1_mu2 /= mu1_sq.to("cuda:4")

    if size_average:
        return mu1_mu2.mean().to('cuda:0')
    else:
        return mu1_mu2.mean(1).mean(1).mean(1).to('cuda:0')

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def toImg(data, renorm_channels = False):
    print("In to toImg: " + str(data.shape))
    if(len(data.shape) == 3):
        im =  cm.coolwarm(data[0])
    elif(len(data.shape) == 4):
        im = toImg(data[:,:,:,int(data.shape[3]/2)], renorm_channels)
    print("Out of toImg: " + str(im.shape))

    return im

def bilinear_interpolate(im, x, y):
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[2]-1)
    x1 = torch.clamp(x1, 0, im.shape[2]-1)
    y0 = torch.clamp(y0, 0, im.shape[3]-1)
    y1 = torch.clamp(y1, 0, im.shape[3]-1)
    
    Ia = im[0, :, x0, y0 ]
    Ib = im[0, :, x1, y0 ]
    Ic = im[0, :, x0, y1 ]
    Id = im[0, :, x1, y1 ]
    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))
    return Ia*wa + Ib*wb + Ic*wc + Id*wd

def trilinear_interpolate(im, x, y, z, device, periodic=False):

    if(device == "cpu"):
        dtype = torch.float
        dtype_long = torch.long
    else:
        dtype = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor

    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    z0 = torch.floor(z).type(dtype_long)
    z1 = z0 + 1
    
    if(periodic):
        x1_diff = x1-x
        x0_diff = 1-x1_diff  
        y1_diff = y1-y
        y0_diff = 1-y1_diff
        z1_diff = z1-z
        z0_diff = 1-z1_diff

        x0 %= im.shape[2]
        y0 %= im.shape[3]
        z0 %= im.shape[4]

        x1 %= im.shape[2]
        y1 %= im.shape[3]
        z1 %= im.shape[4]
        
    else:
        x0 = torch.clamp(x0, 0, im.shape[2]-1)
        x1 = torch.clamp(x1, 0, im.shape[2]-1)
        y0 = torch.clamp(y0, 0, im.shape[3]-1)
        y1 = torch.clamp(y1, 0, im.shape[3]-1)
        z0 = torch.clamp(z0, 0, im.shape[4]-1)
        z1 = torch.clamp(z1, 0, im.shape[4]-1)
        x1_diff = x1-x
        x0_diff = x-x0    
        y1_diff = y1-y
        y0_diff = y-y0
        z1_diff = z1-z
        z0_diff = z-z0
    
    c00 = im[0,:,x0,y0,z0] * x1_diff + im[0,:,x1,y0,z0]*x0_diff
    c01 = im[0,:,x0,y0,z1] * x1_diff + im[0,:,x1,y0,z1]*x0_diff
    c10 = im[0,:,x0,y1,z0] * x1_diff + im[0,:,x1,y1,z0]*x0_diff
    c11 = im[0,:,x0,y1,z1] * x1_diff + im[0,:,x1,y1,z1]*x0_diff

    c0 = c00 * y1_diff + c10 * y0_diff
    c1 = c01 * y1_diff + c11 * y0_diff

    c = c0 * z1_diff + c1 * z0_diff
    return c   

def crop_to_size(frame, max_dim_size):
    x_crop_start = max(frame.shape[2]-max_dim_size, 0)
    x_crop_end = min(frame.shape[2], x_crop_start + max_dim_size)
    
    y_crop_start = max(frame.shape[3]-max_dim_size, 0)
    y_crop_end = min(frame.shape[3], y_crop_start + max_dim_size)
    
    z_crop_start = max(frame.shape[4]-max_dim_size, 0)
    z_crop_end = min(frame.shape[4], z_crop_start + max_dim_size)

    return frame[:,:,x_crop_start:x_crop_end,
    y_crop_start:y_crop_end,z_crop_start:z_crop_end]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def print_to_log_and_console(message, location, file_name):
    #print to console
    print(message)
    #create logs directory if needed
    if not os.path.exists(location):
        try:
            os.makedirs(location)
        except OSError:
            print ("Creation of the directory %s failed" % location)
    #a for append mode, will also create file if it doesn't exist yet
    _file = open(os.path.join(location, file_name), 'a')
    #print to the file
    print(message, file=_file)

def create_folder(start_path, folder_name):
    f_name = folder_name
    full_path = os.path.join(start_path, f_name)
    if not os.path.exists(full_path):
        try:
            os.makedirs(full_path)
        except OSError:
            print_to_log_and_console ("Creation of the directory %s failed" % full_path)
    else:
        #print_to_log_and_console("%s already exists, overwriting save " % (f_name))
        full_path = os.path.join(start_path, f_name)
    if not os.path.exists(full_path):
        try:
            os.makedirs(full_path)
        except OSError:
            print_to_log_and_console ("Creation of the directory %s failed" % full_path)
    return f_name

def AvgPool2D(x : torch.Tensor, size : int):
    with torch.no_grad():
        kernel = torch.ones([size, size]).to(x.device)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, size, size)
        kernel = kernel.repeat(x.shape[1], 1, 1, 1)
        out = F.conv2d(x, kernel, stride=size, padding=0, groups=x.shape[1])
    return out

def AvgPool3D(x : torch.Tensor, size: int):
    with torch.no_grad():
        kernel = torch.ones([size, size, size]).to(x.device)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, size, size, size)
        kernel = kernel.repeat(x.shape[1], 1, 1, 1, 1)
        out = F.conv3d(x, kernel, stride=size, padding=0, groups=x.shape[1])
    return out
