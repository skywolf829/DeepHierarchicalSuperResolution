from __future__ import absolute_import, division, print_function
from utility_functions import create_folder, print_to_log_and_console, weights_init
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from options import *


project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data", "SuperResolutionData")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def save_models(generators, discriminators, opt):
    folder = create_folder(opt["save_folder"], opt["save_name"])
    path_to_save = os.path.join(opt["save_folder"], folder)
    print_to_log_and_console("Saving model to %s" % (path_to_save), 
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    if(opt["save_generators"]):
        gen_states = {}
        
        for i in range(len(generators)):
            gen_states[str(i)] = generators[i].state_dict()
        torch.save(gen_states, os.path.join(path_to_save, "generators"))

    if(opt["save_discriminators"]):
        discrim_states = {}
        for i in range(len(discriminators)):
            discrim_states[str(i)] = discriminators[i].state_dict()
        torch.save(discrim_states, os.path.join(path_to_save, "discriminators"))

    save_options(opt, path_to_save)

def load_models(opt, device):
    generators = []
    discriminators = []
    load_folder = os.path.join(opt["save_folder"], opt["save_name"])

    if not os.path.exists(load_folder):
        print_to_log_and_console("%s doesn't exist, load failed" % load_folder, 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
        return

    from collections import OrderedDict
    if os.path.exists(os.path.join(load_folder, "generators")):
        gen_params = torch.load(os.path.join(load_folder, "generators"),
        map_location=device)
        for i in range(opt["n"]):
            if(str(i) in gen_params.keys()):
                gen_params_compat = OrderedDict()
                for k, v in gen_params[str(i)].items():
                    if("module" in k):
                        gen_params_compat[k[7:]] = v
                    else:
                        gen_params_compat[k] = v
                generator, num_kernels = init_gen(i, opt)
                generator.load_state_dict(gen_params_compat)
                generators.append(generator)

        print_to_log_and_console("Successfully loaded generators", 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    else:
        print_to_log_and_console("Warning: %s doesn't exists - can't load these model parameters" % "generators", 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    if os.path.exists(os.path.join(load_folder, "discriminators")):
        discrim_params = torch.load(os.path.join(load_folder, "discriminators"),
        map_location=device)
        for i in range(opt["n"]):
            if(str(i) in discrim_params.keys()):
                discrim_params_compat = OrderedDict()
                for k, v in discrim_params[str(i)].items():
                    if(k[0:7] == "module."):
                        discrim_params_compat[k[7:]] = v
                    else:
                        discrim_params_compat[k] = v
                discriminator = init_discrim(i, opt)
                discriminator.load_state_dict(discrim_params_compat)
                discriminators.append(discriminator)
        print_to_log_and_console("Successfully loaded discriminators", 
        os.path.join(opt["save_folder"],opt["save_name"]), "log.txt")
    else:
        print_to_log_and_console("Warning: %s doesn't exists - can't load these model parameters" % "s_discriminators", 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    
    return  generators, discriminators

def calc_gradient_penalty(discrim, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1, device=device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    #interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discrim(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def init_scales(opt, dataset):
    ns = []
    if(opt['mode'] == "3D"):
        dims = 3
        ns.append(round(math.log(opt["min_dimension_size"] / dataset.resolution[1]) / math.log(0.5)))
        ns.append(round(math.log(opt["min_dimension_size"] / dataset.resolution[2]) / math.log(0.5)))
        ns.append(round(math.log(opt["min_dimension_size"] / dataset.resolution[3]) / math.log(0.5)))
        res = [dataset.resolution[1], dataset.resolution[2], dataset.resolution[3]]
    else:
        dims = 2
        ns.append(round(math.log(opt["min_dimension_size"] / dataset.resolution[1]) / math.log(0.5)))
        ns.append(round(math.log(opt["min_dimension_size"] / dataset.resolution[2]) / math.log(0.5)))
        res = [dataset.resolution[1], dataset.resolution[2]]
    print(ns)
    opt["n"] = min(ns)
    print("The model will have %i generators" % (opt["n"]))
    for i in range(opt["n"]+1):
        scaling = []
        factor =  0.5**i
        for j in range(dims):
            x = int(res[j] * factor)
            scaling.append(x)
        opt["resolutions"].insert(0,scaling)
    for i in range(opt['n']):
        print("Scale %i: %s -> %s" % (opt["n"] - 1 - i, str(opt["resolutions"][i]), str(opt["resolutions"][i+1])))

def init_gen(scale, opt):
    generator = Generator(opt["resolutions"][scale+1], opt)
    generator.apply(weights_init)

    return generator

def init_discrim(scale, opt):
    discriminator = Discriminator(opt["resolutions"][scale+1], opt)
    discriminator.apply(weights_init)

    return discriminator

class DenseBlock(nn.Module):
    def __init__(self, kernels, growth_channel, opt):
        super(DenseBlock, self).__init__()
        if(opt['mode'] == "2D"):
            conv_layer = nn.Conv2d
        elif(opt['mode'] == "3D"):
            conv_layer = nn.Conv3d
        self.c1 = conv_layer(kernels, growth_channel, kernel_size=opt['kernel_size'],
        stride=opt['stride'],padding=opt['padding'])
        self.c2 = conv_layer(kernels+growth_channel*1, growth_channel, kernel_size=opt['kernel_size'],
        stride=opt['stride'],padding=opt['padding'])
        self.c3 = conv_layer(kernels+growth_channel*2, growth_channel, kernel_size=opt['kernel_size'],
        stride=opt['stride'],padding=opt['padding'])
        self.c4 = conv_layer(kernels+growth_channel*3, growth_channel, kernel_size=opt['kernel_size'],
        stride=opt['stride'],padding=opt['padding'])
        self.lrelu = nn.LeakyReLU(0.2,inplace=True)
        self.final_conv = conv_layer(kernels+growth_channel*4, kernels, kernel_size=opt['kernel_size'],
        stride=opt['stride'],padding=opt['padding'])

    def forward(self,x):       
        c1_out = self.lrelu(self.c1(x))
        c2_out = self.lrelu(self.c2(torch.cat([x, c1_out], 1)))
        c3_out = self.lrelu(self.c3(torch.cat([x, c1_out, c2_out], 1)))
        c4_out = self.lrelu(self.c4(torch.cat([x, c1_out, c2_out, c3_out], 1)))
        final_out = self.final_conv(torch.cat([x, c1_out, c2_out, c3_out, c4_out], 1))
        return final_out

class RRDB(nn.Module):
    def __init__ (self,opt):
        super(RRDB, self).__init__()
        self.db1 = DenseBlock(opt['num_kernels'], int(opt['num_kernels']/4), opt)
        self.db2 = DenseBlock(opt['num_kernels'], int(opt['num_kernels']/4), opt)
        self.db3 = DenseBlock(opt['num_kernels'], int(opt['num_kernels']/4), opt)       
        self.B = torch.tensor([opt['B']])
        self.register_buffer('B_const', self.B)

    def forward(self,x):
        db1_out = self.db1(x) * self.B_const + x
        db2_out = self.db2(db1_out) * self.B_const + db1_out
        db3_out = self.db3(db2_out) * self.B_const + db2_out
        out = db3_out * self.B_const + x
        return out

class Generator(nn.Module):
    def __init__ (self, resolution, opt):
        super(Generator, self).__init__()
        self.resolution = resolution
        self.opt = opt
        if(opt['mode'] == "2D"):
            conv_layer = nn.Conv2d
            self.pix_shuffle = nn.PixelShuffle(2)
        elif(opt['mode'] == "3D"):
            conv_layer = nn.Conv3d

        self.c1 = conv_layer(opt['num_channels'], opt['num_kernels'],
        stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])
        self.blocks = []
        for _ in range(opt['num_blocks']):
            self.blocks.append(RRDB(opt))
        self.blocks =  nn.ModuleList(self.blocks)
        
        self.c2 = conv_layer(opt['num_kernels'], opt['num_kernels'],
        stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])

        # Upscaling happens between 2 and 3
        if(self.opt['mode'] == "2D"):
            fact = 4
        elif(self.opt['mode'] == "3D"):
            fact = 8

        self.c2_vs = conv_layer(opt['num_kernels'], opt['num_kernels']*fact,
            stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])
       
        self.c3 = conv_layer(opt['num_kernels'], opt['num_kernels'],
        stride=opt['stride'],padding=opt['padding'],kernel_size=opt['kernel_size'])

        self.final_conv = conv_layer(opt['num_kernels'], opt['num_channels'],
        stride=opt['stride'],padding=2,kernel_size=5)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.c1(x)
        '''
        out = self.blocks[0](x)
        for i in range(1, len(self.blocks)):
            out = self.blocks[i](out)
        '''
        out = x.clone()
        for i, mod in enumerate(self.blocks):
            out = mod(out)
            
        out = self.c2(out)
        out = x + out

        out = self.c2_vs(out)
        if(self.opt['mode'] == "3D"):
            out = VoxelShuffle(out)
        elif(self.opt['mode'] == "2D"):
            out = self.pix_shuffle(out)
        
        out = self.lrelu(self.c3(out))
        out = self.final_conv(out)
        return out

def VoxelShuffle(t):
    # t has shape [batch, channels, x, y, z]
    # channels should be divisible by 8
    
    input_view = t.contiguous().view(
        1, 2, 2, 2, int(t.shape[1]/8), t.shape[2], t.shape[3], t.shape[4]
    )
    shuffle_out = input_view.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()
    out = shuffle_out.view(
        1, int(t.shape[1]/8), 2*t.shape[2], 2*t.shape[3], 2*t.shape[4]
    )
    return out

class Discriminator(nn.Module):
    def __init__ (self, resolution, opt):
        super(Discriminator, self).__init__()

        self.resolution = resolution

        if(opt['mode'] == "2D" or opt['mode'] == "3Dto2D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
        elif(opt['mode'] == "3D"):
            conv_layer = nn.Conv3d
            batchnorm_layer = nn.BatchNorm3d

        modules = []
        for i in range(opt['num_discrim_blocks']):
            # The head goes from 3 channels (RGB) to num_kernels
            if i == 0:
                modules.append(nn.Sequential(
                    create_conv_layer(conv_layer, opt['num_channels'], opt['num_kernels'], 
                    opt['kernel_size'], opt['stride']),
                    create_batchnorm_layer(batchnorm_layer, opt['num_kernels']),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            # The tail will go from num_kernels to 1 channel for discriminator optimization
            elif i == opt['num_discrim_blocks']-1:  
                tail = nn.Sequential(
                    create_conv_layer(conv_layer, opt['num_kernels'], 1, 
                    opt['kernel_size'], opt['stride'])
                )
                modules.append(tail)
            # Other layers will have 32 channels for the 32 kernels
            else:
                modules.append(nn.Sequential(
                    create_conv_layer(conv_layer, opt['num_kernels'], opt['num_kernels'], 
                    opt['kernel_size'], opt['stride']),
                    create_batchnorm_layer(batchnorm_layer, opt['num_kernels']),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
        self.model =  nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

def create_batchnorm_layer(batchnorm_layer, num_kernels):
    bnl = batchnorm_layer(num_kernels)
    bnl.apply(weights_init)
    return bnl

def create_conv_layer(conv_layer, in_chan, out_chan, kernel_size, stride):
    c = conv_layer(in_chan, out_chan, 
                    kernel_size, stride, 0)
    c.apply(weights_init)
    return c