#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/create_octree.py \
--file HeatedCylinder.h5 --save_name HeatedCylinder_4x.octree \
--epsilon 0.0089 --min_downscaling_level 0 \
--max_downscaling_level 5 --min_chunk 2 \
--device cuda:0 

python -u Code/create_octree.py \
--file HeatedCylinder.h5 --save_name HeatedCylinder_16x.octree \
--epsilon 0.0445 --min_downscaling_level 0 \
--max_downscaling_level 5 --min_chunk 2 \
--device cuda:0 

python -u Code/create_octree.py \
--file HeatedCylinder.h5 --save_name HeatedCylinder_16x_v2.octree \
--epsilon 0.0193 --min_downscaling_level 1 \
--max_downscaling_level 5 --min_chunk 2 \
--device cuda:0 