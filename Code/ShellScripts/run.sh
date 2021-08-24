#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python3 -u train_spatial_SR.py --save_name Isomag2D --train_distributed True --upsample_mode shuffle --beta_1 0.9 \
--num_workers 0 --beta_2 0.999 --data_folder Isomag2D --mode 2D \
--cropping_resolution 256 --patch_size 1024 --training_patch_size 1024 \
--num_blocks 3 --base_num_kernels 96 --epochs 50 --random_flipping True \
--min_dimension_size 16

#python3 -u Code/train.py --save_name Mixing3D --train_distributed True --gpus_per_node 8 \
#--num_workers 0 --data_folder Mixing3D --mode 3D --patch_size 96 --training_patch_size 96 \
#--epochs 100 --min_dimension_size 16 \
#--cropping_resolution 96

#python3 -u Code/train.py --save_name Mixing3D --train_distributed False --gpus_per_node 8 \
#--num_workers 0 --data_folder Mixing3D --mode 3D --patch_size 96 --training_patch_size 96 \
#--epochs 15 --min_dimension_size 32 \
#--cropping_resolution 96 --alpha_2 0.1 