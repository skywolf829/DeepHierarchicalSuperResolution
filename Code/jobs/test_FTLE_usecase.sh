#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python Code/octree_upscaling.py \
    --octree_file HeatedCylinder_16x_v2.octree \
    --volume_file HeatedCylinder.h5 \
    --save_name heatedcylinder_HSR \
    --upscaling_method model \
    --model_name boussinesq_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume True \
    --border_on_octree False \
    --seams False \
    --save_downscaling_levels True \
    --device cuda:0

python Code/octree_upscaling.py \
    --octree_file HeatedCylinder_16x_v2.octree \
    --volume_file HeatedCylinder.h5 \
    --save_name heatedcylinder_nearest \
    --upscaling_method nearest \
    --model_name boussinesq_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --save_downscaling_levels False \
    --seams True \
    --device cuda:0
