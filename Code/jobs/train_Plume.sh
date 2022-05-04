#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py \
--train_distributed true --gpus_per_node 8 \
--mode 3D --model ESRGAN \
--data_folder Plume --save_name plume_ESRGAN_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 5 --num_kernels 32 \
--padding_mode reflect \
--epochs 20 

python -u Code/train.py \
--train_distributed true --gpus_per_node 8 \
--mode 3D --model SSRTVD_NO_G \
--data_folder Plume --save_name plume_SSRTVD_NO_G_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 5 --num_kernels 32 \
--padding_mode reflect \
--epochs 20 

python -u Code/train.py \
--train_distributed true --gpus_per_node 8 \
--mode 3D --model STNet \
--data_folder Plume --save_name plume_STNet_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 5 --num_kernels 32 \
--padding_mode reflect \
--epochs 20 