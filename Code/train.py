from __future__ import absolute_import, division, print_function
import argparse
from datasets import TrainingDataset, SSRTVD_dataset
import datetime
from utility_functions import AvgPool2D, AvgPool3D, print_to_log_and_console, reset_grads, str2bool, toImg
from models import calc_gradient_penalty, init_discrim, init_discrim_t, \
    init_gen, init_scales, load_models, save_models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


def train_single_scale(rank, generators, discriminators, opt, dataset, discriminators_t=None):
    print("Training on device " + str(rank))
    print(discriminators_t)
    if(opt['train_distributed']):        
        print("Initializing process group.")
        opt['device'] = "cuda:" + str(rank)
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://',                                   
            world_size=opt['num_nodes'] * opt['gpus_per_node'],                              
            rank=rank                                               
        )  
    start_t = time.time()

    torch.manual_seed(0)
    
    # Create the new generator and discriminator for this level
    if(len(generators) == opt['scale_in_training']):
        generator = init_gen(len(generators), opt)
        discriminator = init_discrim(len(generators), opt)
        if(opt['model'] == "SSRTVD"):
            discriminator_t = init_discrim_t(len(generators), opt)
    else:
        generator = generators[-1]
        generators.pop(len(generators)-1)
        discriminator = discriminators[-1]
        discriminators.pop(len(discriminators)-1)
        if(opt['model'] == "SSRTVD"):
            discriminator_t = discriminators_t[-1]
            discriminators_t.pop(len(discriminators_t)-1)
    if(opt['model'] == "ESRGAN"):
        combined_models = torch.nn.ModuleList([generator, discriminator]).to(rank)
    elif(opt['model'] == "SSRTVD"):
        combined_models = torch.nn.ModuleList([generator, discriminator, discriminator_t]).to(rank)

    if(opt['train_distributed']):
        combined_models = DDP(combined_models, device_ids=[rank])
        generator = combined_models.module[0]
        discriminator = combined_models.module[1]
        if(opt['model'] == "SSRTVD"):
            discriminator_t = combined_models.module[2]
    else:
        generator = combined_models[0]
        discriminator = combined_models[1]
        if(opt['model'] == "SSRTVD"):
            discriminator_t = combined_models.module[2]
        
    print_to_log_and_console("Training on %s" % (opt["device"]), 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")


    generator_optimizer = optim.Adam(generator.parameters(), lr=opt["g_lr"], 
    betas=(opt["beta_1"],opt["beta_2"]))
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=generator_optimizer,
    milestones=[0.8*opt['epochs']-opt['epoch_number']],gamma=opt['gamma'])

    discriminator_optimizer = optim.Adam(discriminator.parameters(), 
    lr=opt["d_lr"], betas=(opt["beta_1"],opt["beta_2"]))
    discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=discriminator_optimizer,
    milestones=[0.8*opt['epochs']-opt['epoch_number']],gamma=opt['gamma'])
    
    if(opt['model'] == "SSRTVD"):
        discriminator_t_optimizer = optim.Adam(discriminator_t.parameters(), 
            lr=opt["d_lr"], betas=(opt["beta_1"],opt["beta_2"]))
        discriminator_t_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=discriminator_t_optimizer,
            milestones=[0.8*opt['epochs']-opt['epoch_number']],gamma=opt['gamma'])

    if((rank == 0 and opt['train_distributed']) or not opt['train_distributed']):
        writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))
        print(str(len(generators)) + ": " + str(opt["resolutions"][len(generators)]))

    start_time = time.time()
    next_save = 0
    if(opt['train_distributed']):
        volumes_seen = opt['epoch_number'] * int(len(dataset) / opt['gpus_per_node'])
    else:
        volumes_seen = opt['epoch_number'] * len(dataset)

    dataset.set_subsample_dist(int(2**(opt['n']-len(generators)-1)))
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
    
    
    l1_loss = nn.L1Loss().to(opt["device"])
    l2_loss = nn.MSELoss().to(opt['device'])

    for epoch in range(opt['epoch_number'], opt["epochs"]):
        
        for batch_num, real_hr in enumerate(dataloader):
                        
            real_hr = real_hr.to(opt["device"])       
            
            if opt['mode'] == "3D": 
                real_lr = AvgPool3D(real_hr, 2)                
            elif opt['mode'] == "2D":
                real_lr = AvgPool2D(real_hr, 2)

            #print("HR: %s, LR: %s" % (real_hr.shape, real_lr.shape))
            D_loss = 0
            G_loss = 0        
            rec_loss = 0        
            
            if(opt["alpha_2"] > 0.0 and (opt['model'] == "SSRTVD" or \
                (opt['model'] == "ESRGAN" and epoch > opt['epochs']/2))):
                # Update spatial discrim
                for _ in range(opt["discriminator_steps"]):
                    discriminator.zero_grad()
                    generator.zero_grad()
                    D_loss = 0
                    
                    output_real = discriminator(
                        real_hr[:,1:2] if opt['model'] == "SSRTVD" else real_hr)

                    fake = generator(
                        real_lr[:,1:2] if opt['model'] == "SSRTVD" else real_lr).detach()
                    output_fake = discriminator(fake)
                    
                    # Relativistic discriminator
                    if(opt['model'] == "ESRGAN"):
                        D_loss = -torch.log(torch.sigmoid(output_real.mean() - output_fake.mean())) - \
                            torch.log(1-torch.sigmoid(output_fake.mean() - output_real.mean()))
                    elif(opt['model'] == "SSRTVD"):
                        D_loss = ((output_real.mean() - 1)**2 + output_fake.mean()**2)/2

                    D_loss.backward(retain_graph=True)
                    discriminator_optimizer.step()
                
                if opt['model'] == "SSRTVD":
                    for _ in range(opt["discriminator_steps"]):
                        discriminator_t.zero_grad()
                        generator.zero_grad()
                        D_T_loss = 0
                        
                        output_real = discriminator_t(real_hr)

                        fake = generator(real_lr.transpose(0,1)).transpose(0,1).detach()
                        output_fake = discriminator_t(fake)

                        D_T_loss = ((output_real.mean() - 1)**2 + output_fake.mean()**2)/2
                        
                        D_T_loss.backward(retain_graph=True)
                        discriminator_t_optimizer.step()


            # Update generator
            for _ in range(opt["generator_steps"]):
                generator.zero_grad()
                discriminator.zero_grad()
                if opt['model'] == "SSRTVD":
                    discriminator_t.zero_grad()
                G_loss = 0
                
                if opt['model'] == "SSRTVD":
                    fake = generator(real_lr.transpose(0, 1)).transpose(0,1)
                else:
                    fake = generator(real_lr)
                
                if(opt['alpha_1'] > 0.0):
                    rec_loss = l1_loss(fake, real_hr) if opt['model'] == "ESRGAN" else \
                        l2_loss(fake[:1:2], real_hr) * opt["alpha_1"]
                    G_loss += rec_loss
                    rec_loss = rec_loss.item()

                if(opt["alpha_2"] > 0.0 and (opt['model'] == "SSRTVD" or \
                    (opt['model'] == "ESRGAN" and epoch > opt['epochs']/2))):            
                    output = discriminator(fake[:,1:2] if opt['model'] == "SSRTVD" else fake)
                    
                    if opt['model'] == "SSRTVD":
                        
                        # feature loss
                        real_feat_maps = discriminator_t.feature_maps(real_hr)
                        fake_feat_maps = discriminator_t.feature_maps(fake)
                        feat_loss = 0
                        for feat_map in range(len(real_feat_maps)):
                            N_k = 1
                            for dim in real_feat_maps[feat_map].shape:
                                N_k *= dim
                            feat_loss += l2_loss(real_feat_maps[feat_map], 
                                fake_feat_maps[feat_map]) / N_k


                        # temporal discrim loss
                        d_t_loss = (discriminator_t(fake).mean() - 1)**2

                        # spatial discrim loss
                        d_s_loss = (discriminator(fake[:,1:2]).mean() - 1)**2

                        adv_G_loss = (d_t_loss + d_s_loss) * opt['alpha_2']
                        G_loss += feat_loss * 0.05 + (d_t_loss + d_s_loss) * opt['alpha_2']
                    else:
                        # spatial relativistic loss
                        output_real_discrim = discriminator(real_hr).detach()
                        adv_G_loss = torch.log(1 - torch.sigmoid(output_real_discrim.mean() - output.mean())) - \
                            torch.log(torch.sigmoid(output.mean() - output_real_discrim.mean()))

                    G_loss += adv_G_loss                    
                    gen_adv_err = adv_G_loss.item()
                
                G_loss.backward(retain_graph=True)
                generator_optimizer.step()

            volumes_seen += 1

            if(((rank == 0 and opt['train_distributed']) or not opt['train_distributed'])):
                if(volumes_seen % 50 == 0 and opt['model'] == "ESRGAN"):
                    rec_numpy = fake.detach().cpu().numpy()[0]
                    rec_cm = toImg(rec_numpy)

                    writer.add_image("reconstructed/%i"%len(generators), 
                        rec_cm, volumes_seen, dataformats="HWC")

                    real_numpy = real_hr.detach().cpu().numpy()[0]
                    real_cm = toImg(real_numpy)
                    
                    writer.add_image("real/%i"%len(generators), 
                        real_cm, volumes_seen, dataformats="HWC")

                num_total = opt['epochs']*len(dataset)
                if(opt['train_distributed']):
                    num_total = int(num_total / (opt['num_nodes'] * opt['gpus_per_node']))
                print_to_log_and_console("%i/%i: Dloss=%.02f Gloss=%.02f L1=%.04f" %
                    (volumes_seen, num_total, D_loss, G_loss, rec_loss), 
                os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
                
                if(opt['alpha_1'] > 0):
                    writer.add_scalar('L1/%i'%len(generators), rec_loss, volumes_seen)

                if(opt["alpha_2"] > 0.0 and (opt['model'] == "SSRTVD" or \
                    (opt['model'] == "ESRGAN" and epoch > opt['epochs']/2))):
                    writer.add_scalar('D_loss_scale/%i'%len(generators), D_loss.item(), volumes_seen)
                    if(opt['model'] == "SSRTVD"):
                        writer.add_scalar('D_T_loss_scale/%i'%len(generators), d_t_loss.item(), volumes_seen)    
                    writer.add_scalar('G_loss_scale/%i'%len(generators), gen_adv_err, volumes_seen) 

                    

                if(volumes_seen % opt['save_every'] == 0):
                    opt["iteration_number"] = batch_num
                    opt["epoch_number"] = epoch
                    save_models(generators + [generator], 
                        discriminators + [discriminator], opt,
                        discriminators_t + [discriminator_t] if opt['model'] == "SSRTVD" else None)
                    
        if(rank == 0):
            print("Epoch done")
        if(opt["alpha_2"] > 0.0 and (opt['model'] == "SSRTVD" or \
                    (opt['model'] == "ESRGAN" and epoch > opt['epochs']/2))):
            discriminator_scheduler.step()
            if(opt['model'] == "SSRTVD"):
                discriminator_t_scheduler.step()
        generator_scheduler.step()
        if(rank == 0):
            print("Step")


    generator = reset_grads(generator, False)
    generator.eval()
    discriminator = reset_grads(discriminator, False)
    discriminator.eval()
    if opt['model'] == "SSRTVD":
        discriminator_t = reset_grads(discriminator_t, False)
        discriminator_t.eval()

    if(not opt['train_distributed'] or rank == 0):
        save_models(generators + [generator], 
            discriminators + [discriminator], opt,
            discriminators_t + [discriminator_t] if opt['model'] == "SSRTVD" else None)
    if not opt['train_distributed']:
        if(opt['model'] == "ESRGAN"):
            return generator, discriminator
        else:
            return generator, discriminator, discriminator_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on an input that is 2D')

    parser.add_argument('--mode',default=None,type=str,help='The type of input - 2D, 3D')
    parser.add_argument('--model',default=None,type=str,help='Model to use, ESRGAN or SSRTVD')
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


    if(args['load_from'] is None):
        # Init models
        generators = []
        discriminators = []
        discriminators_t = []
        opt = Options.get_default()

        # Read arguments and update our options
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]
         # Determine scales    
        if(opt['model'] == "ESRGAN"):
            dataset = TrainingDataset(opt)
        elif(opt['model'] == "SSRTVD"):
            dataset = SSRTVD_dataset(opt)
        init_scales(opt, dataset)
    else:        
        opt = load_options(os.path.join(save_folder, args["load_from"]))
        opt["device"] = args["device"]
        opt["save_name"] = args["load_from"]
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]
        if(opt['model'] == "ESRGAN"):
            dataset = TrainingDataset(opt)
            generators, discriminators = load_models(opt,args["device"])
        elif(opt['model'] == "SSRTVD"):
            dataset = SSRTVD_dataset(opt)            
            generators, discriminators, discriminators_t = load_models(opt,args["device"])

    now = datetime.datetime.now()
    start_time = time.time()
    print_to_log_and_console("Started training at " + str(now), 
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    # Train each scale 1 by 1
    i = opt['scale_in_training']
    while i < opt["n"]:

        start_time_scale_n = time.time()

        print_to_log_and_console(str(datetime.datetime.now()) + " - Beginning training on scale " + str(i),
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

        if(opt['train_distributed']):
            os.environ['MASTER_ADDR'] = '127.0.0.1'              
            os.environ['MASTER_PORT'] = '29500' 
            mp.spawn(train_single_scale,
                args=(generators, discriminators, opt, dataset, 
                    discriminators_t if opt['model'] == "SSRTVD" else None),
                nprocs=opt['gpus_per_node'],
                join=True)
        else:
            if(opt['model'] == "ESRGAN"):
                generator, discriminator = train_single_scale(opt['device'], generators, 
                    discriminators, opt, dataset, None)
            else:
                generator, discriminator, discriminator_t = train_single_scale(opt['device'], generators, 
                    discriminators, opt, dataset,
                    discriminators_t)

        if(opt['model'] == "ESRGAN"):
            generators, discriminators = load_models(opt,opt["device"])
        elif(opt['model'] == "SSRTVD"): 
            generators, discriminators, discriminators_t = load_models(opt,args["device"])
        
        i += 1
        opt['scale_in_training'] += 1
        opt['iteration_number'] = 0
        opt['epoch_number'] = 0

            
        time_passed = (time.time() - start_time_scale_n) / 60
        print_to_log_and_console("%s - Finished training in scale %i in %f minutes" % (str(datetime.datetime.now()), len(generators)-1, time_passed),
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt") 
        


    time_passed = (time.time() - start_time) / 60
    print_to_log_and_console("%s - Finished training  in %f minutes" % (str(datetime.datetime.now()), time_passed),
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt") 
    save_models(generators, discriminators, opt, 
        discriminators_t if opt['model'] == "SSRTVD" else None)



