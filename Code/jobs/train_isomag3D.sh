#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/train.py \
--train_distributed false --gpus_per_node 8 \
--mode 3D --model ESRGAN \
--data_folder Isomag3D --save_name isomag3D_ESRGAN \
--min_dimension_size 32 --cropping_resolution 96 \
--num_blocks 5 --num_kernels 32 \
--padding_mode reflect \
--epochs 20 