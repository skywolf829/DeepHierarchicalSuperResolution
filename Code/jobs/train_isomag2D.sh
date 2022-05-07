#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py \
--train_distributed true --gpus_per_node 8 \
--mode 2D --model ESRGAN \
--data_folder Isomag2D --save_name isomag2D_ESRGAN_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 5 --num_kernels 32 \
--padding_mode reflect \
--epochs 100 

python -u Code/train.py \
--train_distributed true --gpus_per_node 8 \
--mode 2D --model STNet \
--data_folder Isomag2D --save_name isomag2D_STNet_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 5 --num_kernels 32 \
--padding_mode reflect \
--epochs 100 

python -u Code/train.py \
--train_distributed true --gpus_per_node 8 \
--mode 2D --model SSRTVD_NO_G \
--data_folder Isomag2D --save_name isomag2D_SSRTVD_NO_G_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 5 --num_kernels 32 \
--padding_mode reflect \
--epochs 100 

