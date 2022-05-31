#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python Code/octree_upscaling.py \
    --octree_file Plume_64xreduction.octree \
    --volume_file Plume.h5 \
    --save_name plume_STNet_HSR \
    --upscaling_method model \
    --model_name plume_STNet_new \
    --save_original_volume True \
    --save_error_volume True \
    --seams False \
    --device cuda:0

python Code/octree_upscaling.py \
    --octree_file Plume_64xreduction.octree \
    --volume_file Plume.h5 \
    --save_name plume_nearest \
    --upscaling_method nearest \
    --model_name plume_STNet_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams True \
    --device cuda:0

python Code/uniform_upscaling.py \
    --volume_file Plume.h5 \
    --save_name plume_STNet_uniform \
    --scale_factor 4 \
    --upscaling_method model \
    --model_name plume_STNet_new \
    --save_error_volume True \
    --device cuda:0

    ################################

python Code/octree_upscaling.py \
    --octree_file Plume_psnrcomparison.octree \
    --volume_file Plume.h5 \
    --save_name plume_STNet_psnrcomparison_HSR \
    --upscaling_method model \
    --model_name plume_STNet_new \
    --save_original_volume True \
    --save_error_volume True \
    --seams False \
    --device cuda:0

python Code/octree_upscaling.py \
    --octree_file Plume_psnrcomparison.octree \
    --volume_file Plume.h5 \
    --save_name plume_psnrcomparison_nearest \
    --upscaling_method nearest \
    --model_name plume_STNet_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams True \
    --device cuda:0
