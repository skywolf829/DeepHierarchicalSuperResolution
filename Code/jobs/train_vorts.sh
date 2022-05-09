#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py \
--train_distributed true --gpus_per_node 8 \
--mode 3D --model ESRGAN \
--data_folder Vorts --save_name vorts_ESRGAN_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 5 --num_kernels 32 \
--padding_mode reflect \
--epochs 250 

python -u Code/train.py \
--train_distributed true --gpus_per_node 8 \
--mode 3D --model STNet \
--data_folder Vorts --save_name vorts_STNet_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 5 --num_kernels 32 \
--padding_mode reflect \
--epochs 250 

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model SSRTVD_NO_D \
--data_folder Vorts --save_name vorts_SSRTVD_NO_D_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 5 --num_kernels 32 \
--padding_mode reflect \
--epochs 250 

