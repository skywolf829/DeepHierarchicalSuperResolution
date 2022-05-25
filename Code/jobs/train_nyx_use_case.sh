#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model ESRGAN \
--g_lr 0.00002 \
--data_folder nyx64 --save_name nyx_use_case \
--min_dimension_size 64 --cropping_resolution 76 \
--num_blocks 3 --num_kernels 96 \
--nyx_use_case True \
--training_patch_size 76 --patch_size 76 \
--padding_mode reflect \
--epochs 100 --device cuda:0 