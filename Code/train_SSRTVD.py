from __future__ import absolute_import, division, print_function
import argparse
from datasets import TrainingDataset
import datetime
from utility_functions import AvgPool2D, AvgPool3D, print_to_log_and_console, reset_grads, str2bool, toImg
from models import save_SSRTVD_models, SSRTVD_D_S, SSRTVD_D_T, SSRTVD_G
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
from options import *
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp


project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data", "SuperResolutionData")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")


def train(rank, generator, discriminator_s, discriminator_t, opt, dataset):
    print("Training on device " + str(rank))
    if(opt['train_distributed']):        
        print("Initializing process group.")
        opt['device'] = "cuda:" + str(rank)
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://',                                   
            world_size=opt['num_nodes'] * opt['gpus_per_node'],                              
            rank=rank                                               
        )  
    torch.manual_seed(0)
    

    combined_models = torch.nn.ModuleList([generator, discriminator_s, discriminator_t]).to(rank)
    if(opt['train_distributed']):
        combined_models = DDP(combined_models, device_ids=[rank])
        generator = combined_models.module[0]
        discriminator_s = combined_models.module[1]
        discriminator_t = combined_models.module[2]
    else:
        generator = combined_models[0]
        discriminator_s = combined_models[1]
        discriminator_t = combined_models[2]
        
    print_to_log_and_console("Training on %s" % (opt["device"]), 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")


    generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4, 
        betas=(0.5,0.999))

    discriminator_s_optimizer = optim.Adam(discriminator_s.parameters(), 
        lr=4e-4, betas=(0.5,0.999))

    discriminator_t_optimizer = optim.Adam(discriminator_t.parameters(), 
        lr=4e-4, betas=(0.5,0.999))
    
    if((rank == 0 and opt['train_distributed']) or not opt['train_distributed']):
        writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))

    start_time = time.time()
    next_save = 0
    volumes_seen = 0

    dataset.set_subsample_dist(4)
    if(opt['train_distributed']):
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, 
            num_replicas=opt['num_nodes']*opt['gpus_per_node'],rank=rank)
        dataloader = torch.utils.data.DataLoader(
            batch_size=1,
            dataset=dataset,
            shuffle=False,
            num_workers=opt["num_workers"],
            pin_memory=True,
            sampler=train_sampler
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            batch_size=1,
            dataset=dataset,
            shuffle=True,
            num_workers=opt["num_workers"],
            pin_memory=True
        )
    
    
    loss = nn.MSELoss().to(opt["device"])

    for epoch in range(0, opt["epochs"]):
                
        for batch_num, real_hr in enumerate(dataloader):
                        
            real_hr = real_hr.to(opt["device"])       
            
            if opt['mode'] == "3D": 
                real_lr = AvgPool3D(real_hr, 4)                
            elif opt['mode'] == "2D":
                real_lr = AvgPool2D(real_hr, 4)

            #print("HR: %s, LR: %s" % (real_hr.shape, real_lr.shape))
            G_loss = 0        
            D_S_loss = 0
            D_T_loss = 0
            rec_loss = 0        
            
            # Update discriminator_s
            for _ in range(2):
                discriminator_s.zero_grad()
                discriminator_t.zero_grad()
                generator.zero_grad()
                D_S_loss = 0
                
                output_real = discriminator_s(real_hr[:,1:2])
                D_S_loss -= output_real.mean()

                fake = generator(real_lr[:,1:2]).detach()
                output_fake = discriminator_s(fake)
                D_S_loss += output_fake.mean()
                
                D_S_loss.backward(retain_graph=True)
                discriminator_s_optimizer.step()
            
            # Update discriminator_t
            for _ in range(2):
                discriminator_s.zero_grad()
                discriminator_t.zero_grad()
                generator.zero_grad()
                D_T_loss = 0
                
                output_real = discriminator_t(real_hr)
                D_T_loss -= output_real.mean()

                fake = generator(real_lr.transpose(0,1)).transpose(0,1).detach()
                output_fake = discriminator_t(fake)
                D_T_loss += output_fake.mean()
                
                D_T_loss.backward(retain_graph=True)
                discriminator_t_optimizer.step()


            # Update generator: maximize D(G(z))
            for _ in range(1):
                generator.zero_grad()
                discriminator_s.zero_grad()
                discriminator_t.zero_grad()
                G_loss = 0
                
                fake = generator(real_lr.transpose(0, 1)).transpose(0,1)

                rec_loss = loss(fake[:,1:2], real_hr[:,1:2]) * 1 #lambda_2
                G_loss += rec_loss
                rec_loss = rec_loss.item()
                        
                d_s_loss = discriminator_s(fake[:,1:2])
                G_loss += (-d_s_loss * 1e-3) #lambda_1
                gen_adv_err = -d_s_loss.item()

                d_t_loss = discriminator_t(fake)
                G_loss += (-d_t_loss * 1e-3) #lambda_1
                gen_adv_err += -d_t_loss.item()

                
                G_loss.backward(retain_graph=True)
                generator_optimizer.step()

            volumes_seen += 1

            if(((rank == 0 and opt['train_distributed']) or not opt['train_distributed'])):
                num_total = opt['epochs']*len(dataset)
                if(opt['train_distributed']):
                    num_total = int(num_total / (opt['num_nodes'] * opt['gpus_per_node']))
                
                print_to_log_and_console("%i/%i: D_S_loss=%.02f D_T_loss=%.02f Gloss=%.02f L1=%.04f" %
                    (volumes_seen, num_total, D_S_loss, D_T_loss, G_loss, rec_loss), 

                os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
                
                writer.add_scalar('L1/%i'%len(generators), rec_loss, volumes_seen)                
                writer.add_scalar('D_S_loss/%i'%len(generators), D_S_loss.item(), volumes_seen)
                writer.add_scalar('D_T_loss/%i'%len(generators), D_T_loss.item(), volumes_seen)      
                writer.add_scalar('G_loss/%i'%len(generators), gen_adv_err, volumes_seen) 
                
                if(volumes_seen % opt['save_every'] == 0):
                    opt["iteration_number"] = batch_num
                    opt["epoch_number"] = epoch
                    save_SSRTVD_models(generator, discriminator_s, discriminator_t, opt)
                    
        if(rank == 0):
            print("Epoch done")


    if(not opt['train_distributed'] or rank == 0):
        save_SSRTVD_models(generator, discriminator_s, discriminator_t, opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on an input that is 2D')

    parser.add_argument('--mode',default=None,help='The type of input - 2D, 3D')
    parser.add_argument('--data_folder',default=None,type=str,help='File to train on')
    parser.add_argument('--save_folder',default=None, help='The folder to save the models folder into')
    parser.add_argument('--save_name',default=None, help='The name for the folder to save the model')
    parser.add_argument('--num_channels',default=None,type=int,help='Number of channels to use')
    parser.add_argument('--min_dimension_size',default=None,type=int,help='Minimum dimension size')
    parser.add_argument('--cropping_resolution',default=None,type=int,help='Res to crop')

    parser.add_argument('--num_workers',default=None, type=int,help='Number of workers for dataset loader')
    parser.add_argument('--random_flipping',default=None, type=str2bool,help='Data augmentation')
   
    parser.add_argument('--num_blocks',default=None,type=int, help='Num of conv-batchnorm-relu blocks per gen/discrim')
    parser.add_argument('--num_discrim_blocks',default=None,type=int, help='Num of conv-batchnorm-relu blocks per gen/discrim')
    parser.add_argument('--num_kernels',default=None,type=int, help='Num conv kernels in lowest layer')
    parser.add_argument('--kernel_size',default=None, type=int,help='Conv kernel size')    
    parser.add_argument('--padding',default=None, type=int,help='Conv padding')
    parser.add_argument('--stride',default=None, type=int,help='Conv stride length')
    parser.add_argument('--B',default=None, type=float,help='Residual scaling factor')
            
    parser.add_argument('--train_distributed',type=str2bool,default=None, help='Use distributed training')
    parser.add_argument('--device',type=str,default=None, help='Device to use')
    parser.add_argument('--gpus_per_node',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--num_nodes',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--ranking',default=None, type=int,help='Whether or not to save discriminators')

    parser.add_argument('--save_generators',default=None, type=str2bool,help='Whether or not to save generators')
    parser.add_argument('--save_discriminators',default=None, type=str2bool,help='Whether or not to save discriminators')
    parser.add_argument('--patch_size',default=None, type=int,help='Patch size for reconstruction')
    parser.add_argument('--training_patch_size',default=None, type=int,help='Training patch size')

    parser.add_argument('--alpha_1',default=None, type=float,help='Reconstruction loss coefficient')
    parser.add_argument('--alpha_2',default=None, type=float,help='Adversarial loss coefficient')
    
    parser.add_argument('--generator_steps',default=None, type=int,help='Number of generator steps to take')
    parser.add_argument('--discriminator_steps',default=None, type=int,help='Number of discriminator steps to take')
    parser.add_argument('--epochs',default=None, type=int,help='Number of epochs to use')
    parser.add_argument('--minibatch',default=None, type=int,help='Size of minibatch to train on')
    parser.add_argument('--g_lr',default=None, type=float,help='Learning rate for the generator')    
    parser.add_argument('--d_lr',default=None, type=float,help='Learning rate for the discriminator')
    parser.add_argument('--beta_1',default=None, type=float,help='')
    parser.add_argument('--beta_2',default=None, type=float,help='')
    parser.add_argument('--gamma',default=None, type=float,help='')

    parser.add_argument('--load_from',default=None, type=str,help='Load a model to continue training')
    parser.add_argument('--save_every',default=None, type=int,help='How often to save during training')


    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data", "SuperResolutionData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")


    
    opt = Options.get_default()

    # Read arguments and update our options
    for k in args.keys():
        if args[k] is not None:
            opt[k] = args[k]
        
    # Init models
    generator = SSRTVD_G(opt)
    discriminator_s = SSRTVD_D_S(opt)
    discriminator_t = SSRTVD_D_T(opt)

    dataset = TrainingDataset(opt)
    

    now = datetime.datetime.now()
    start_time = time.time()
    print_to_log_and_console("Started training at " + str(now), 
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    # Train each scale 1 by 1
    

    start_time_scale_n = time.time()

    print_to_log_and_console(str(datetime.datetime.now()) + " - Beginning training on scale " + str(i),
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    if(opt['train_distributed']):
        os.environ['MASTER_ADDR'] = '127.0.0.1'              
        os.environ['MASTER_PORT'] = '29500' 
        mp.spawn(train,
            args=(generator, discriminator_s, discriminator_t, 
                opt, dataset),
            nprocs=opt['gpus_per_node'],
            join=True)
    else:
        train(opt['device'], generator, discriminator_s, 
            discriminator_t, opt, dataset)

    generators, discriminators = load_models(opt,opt["device"])
    

    time_passed = (time.time() - start_time) / 60
    print_to_log_and_console("%s - Finished training  in %f minutes" % \
        (str(datetime.datetime.now()), time_passed),
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt") 
    save_models(generators, discriminators, opt)



