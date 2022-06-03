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

python -u Code/create_octree.py \
--file HeatedCylinder2.h5 --save_name HeatedCylinder2_5x.octree \
--epsilon 0.11 --min_downscaling_level 0 \
--max_downscaling_level 5 --min_chunk 2 \
--device cuda:0 

python -u Code/create_octree.py \
--file HeatedCylinder3.h5 --save_name HeatedCylinder3_10x.octree \
--epsilon 0.035 --min_downscaling_level 1 \
--max_downscaling_level 5 --min_chunk 2 \
--device cuda:0 

python -u Code/create_octree.py \
--file HeatedCylinder4.h5 --save_name HeatedCylinder4_10x.octree \
--epsilon 0.044 --min_downscaling_level 1 \
--max_downscaling_level 5 --min_chunk 2 \
--device cuda:0 

python -u Code/create_octree.py \
--file Nyx.h5 --save_name Nyx.octree \
--epsilon 0.1 --min_downscaling_level 0 \
--max_downscaling_level 6 --min_chunk 8 \
--device cuda:0 

python -u Code/create_octree.py \
--file Mixing3D.h5 --save_name Mixing.octree \
--epsilon 0.1 --min_downscaling_level 0 \
--max_downscaling_level 6 --min_chunk 16 \
--device cuda:0 

python -u Code/create_octree.py \
--file HeatedCylinder5.h5 --save_name HeatedCylinder5.octree \
--epsilon 0.02 --min_downscaling_level 1 \
--max_downscaling_level 5 --min_chunk 2 \
--device cuda:0 

python -u Code/create_octree.py \
--file HeatedCylinder6.h5 --save_name HeatedCylinder6.octree \
--epsilon 0.02 --min_downscaling_level 1 \
--max_downscaling_level 5 --min_chunk 2 \
--device cuda:0 