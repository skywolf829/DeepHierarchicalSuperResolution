#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

#python3 -u Code/train.py --save_name Isomag2D --train_distributed True \
#--beta_1 0.9 --beta_2 0.999 \
#--num_workers 0 --data_folder Isomag2D --mode 2D \
#--cropping_resolution 256 --patch_size 1024 --training_patch_size 1024 \
#--num_blocks 3 --epochs 50 --random_flipping True \
#--min_dimension_size 16

python3 -u Code/train.py --save_name Isomag3D --train_distributed True \
--beta_1 0.9 --beta_2 0.999 \
--num_workers 0 --data_folder Isomag3D --mode 3D \
--cropping_resolution 96 --patch_size 96 --training_patch_size 96 \
--num_blocks 3 --epochs 50 --random_flipping True \
--min_dimension_size 128

#python3 -u Code/train.py --save_name Mixing3D --train_distributed True --gpus_per_node 8 \
#--num_workers 0 --data_folder Mixing3D --mode 3D --patch_size 96 --training_patch_size 96 \
#--epochs 100 --min_dimension_size 16 \
#--cropping_resolution 96
