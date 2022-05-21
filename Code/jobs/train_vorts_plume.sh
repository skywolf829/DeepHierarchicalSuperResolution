#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model ESRGAN \
--g_lr 0.00002 \
--data_folder Vorts --save_name vorts_ESRGAN_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 3 --num_kernels 96 \
--training_patch_size 96 --patch_size 96 \
--padding_mode reflect \
--epochs 400 --device cuda:0 &

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model STNet \
--g_lr 0.00002 \
--data_folder Vorts --save_name vorts_STNet_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 3 --num_kernels 96 \
--training_patch_size 96 --patch_size 96 \
--padding_mode reflect \
--epochs 400 --device cuda:1 &

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model SSRTVD_NO_D \
--g_lr 0.00002 \
--data_folder Vorts --save_name vorts_SSRTVD_NO_D_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 3 --num_kernels 96 \
--training_patch_size 96 --patch_size 96 \
--padding_mode reflect \
--epochs 400 --device cuda:2 &

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model ESRGAN \
--g_lr 0.0001 \
--data_folder Plume --save_name plume_ESRGAN_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 3 --num_kernels 96 \
--training_patch_size 96 --patch_size 96 \
--padding_mode reflect \
--epochs 750 --device cuda:3 & 

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model STNet \
--g_lr 0.00001 \
--data_folder Plume --save_name plume_STNet_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 3 --num_kernels 96 \
--training_patch_size 96 --patch_size 96 \
--padding_mode reflect \
--epochs 750 --device cuda:4 &

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model SSRTVD_NO_D \
--g_lr 0.00005 \
--data_folder Plume --save_name plume_SSRTVD_NO_D_new \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 3 --num_kernels 96 \
--training_patch_size 96 --patch_size 96 \
--padding_mode reflect \
--epochs 750 --device cuda:5 &

wait