#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python3 -u Code/train.py --save_name Isomag2D_ESRGAN --train_distributed True \
--beta_1 0.9 --beta_2 0.999 \
--num_workers 0 --data_folder Isomag2D --mode 2D \
--cropping_resolution 256 --patch_size 1024 --training_patch_size 1024 \
--num_blocks 3 --epochs 50 --random_flipping true \
--min_dimension_size 32 --g_lr 0.0002 --d_lr 0.0002 --alpha_1 1.0 --alpha_2 0.00 \
--model ESRGAN --generator_steps 1 --discriminator_steps 0

#python3 -u Code/train.py --save_name Isomag3D_noGAN --train_distributed True \
#--beta_1 0.9 --beta_2 0.999 \
#--num_workers 0 --data_folder Isomag3D --mode 3D \
#--cropping_resolution 96 --patch_size 96 --training_patch_size 96 \
#--num_blocks 3 --epochs 50 --random_flipping True \
#--min_dimension_size 64 --alpha_2 0.0

#python3 -u Code/train.py --save_name Mixing3D_noGAN --train_distributed True --gpus_per_node 8 \
#--num_workers 0 --data_folder Mixing3D --mode 3D --patch_size 96 --training_patch_size 96 \
#--epochs 100 --min_dimension_size 16 \
#--cropping_resolution 96 --alpha_2 0

#python3 -u Code/train.py --save_name Mixing3D --train_distributed True --gpus_per_node 8 \
#--num_workers 0 --data_folder Mixing3D --mode 3D --patch_size 96 --training_patch_size 96 \
#--epochs 100 --min_dimension_size 16 \
#--cropping_resolution 96 --alpha_2 0.1

#python3 -u Code/train.py --save_name Plume_noGAN --train_distributed True --gpus_per_node 8 \
#--num_workers 0 --data_folder Plume --mode 3D --patch_size 96 --training_patch_size 96 \
#--epochs 100 --min_dimension_size 16 \
#--cropping_resolution 96 --alpha_2 0.0

#python3 -u Code/train.py --save_name Vorts --train_distributed True --gpus_per_node 8 \
#--num_workers 0 --data_folder Vorts --mode 3D --patch_size 96 --training_patch_size 96 \
#--epochs 100 --min_dimension_size 16 \
#--cropping_resolution 96

#python3 -u Code/train.py --save_name Vorts_noGAN --train_distributed True --gpus_per_node 8 \
#--num_workers 0 --data_folder Vorts --mode 3D --patch_size 96 --training_patch_size 96 \
#--epochs 100 --min_dimension_size 16 \
#--cropping_resolution 96 --alpha_2 0.0