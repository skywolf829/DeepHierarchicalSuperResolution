#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 2D --model SSRTVD_NO_D \
--g_lr 0.00002 \
--data_folder boussinesq --save_name boussinesq_SSRTVD_NO_D_new \
--min_dimension_size 32 --cropping_resolution 256 \
--num_blocks 3 --num_kernels 96 \
--patch_size 1024 --training_patch_size 1024 \
--padding_mode reflect \
--epochs 4 --device cuda:0

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 2D --model ESRGAN \
--g_lr 0.00002 \
--data_folder boussinesq --save_name boussinesq_ESRGAN_new \
--min_dimension_size 32 --cropping_resolution 256 \
--num_blocks 3 --num_kernels 96 \
--patch_size 1024 --training_patch_size 1024 \
--padding_mode reflect \
--epochs 4 --device cuda:0

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 2D --model STNet \
--g_lr 0.00002 \
--data_folder boussinesq --save_name boussinesq_STNet_new \
--min_dimension_size 32 --cropping_resolution 256 \
--num_blocks 3 --num_kernels 96 \
--patch_size 1024 --training_patch_size 1024 \
--padding_mode reflect \
--epochs 4 --device cuda:0