#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/create_octree.py \
--file Isomag2D.h5 --save_name Isomag2D_2x.octree \
--epsilon 0.1 --min_downscaling_level 0 \
--max_downscaling_level 3 --min_chunk 2 \
--device cuda:0 

