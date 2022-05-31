#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model ESRGAN \
--g_lr 0.00002 --d_lr 0.00002 \
--alpha_2 0.5 --discriminator_steps 1 \
--data_folder nyx64 --save_name nyx_use_case_GAN \
--num_blocks 3 --num_kernels 96 \
--nyx_use_case True \
--min_dimension_size 64 --cropping_resolution 64 \
--training_patch_size 64 --patch_size 64 \
--padding_mode reflect \
--epochs 100 --device cuda:0 