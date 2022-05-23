#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model ESRGAN \
--g_lr 0.00002 \
--data_folder Supernova --save_name supernova_ESRGAN_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 3 --num_kernels 96 \
--training_patch_size 96 --patch_size 96 \
--padding_mode reflect \
--epochs 100 --device cuda:0 &

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model STNet \
--g_lr 0.00002 \
--data_folder Supernova --save_name supernova_STNet_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 3 --num_kernels 96 \
--training_patch_size 96 --patch_size 96 \
--padding_mode reflect \
--epochs 100 --device cuda:1 &

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model SSRTVD_NO_D \
--g_lr 0.00002 \
--data_folder Supernova --save_name supernova_SSRTVD_NO_D_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 3 --num_kernels 96 \
--training_patch_size 96 --patch_size 96 \
--padding_mode reflect \
--epochs 100 --device cuda:2 &

wait