#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python Code/octree_upscaling.py \
    --octree_file Isomag2D_4xreduction.octree \
    --volume_file Isomag2D.h5 \
    --save_name Isomag2D_4xreduction_HSR_noseams \
    --upscaling_method model \
    --model_name Isomag2D_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams False \
    --device cuda:0

python Code/octree_upscaling.py \
    --octree_file Isomag2D_4xreduction.octree \
    --volume_file Isomag2D.h5 \
    --save_name Isomag2D_4xreduction_nearest \
    --upscaling_method nearest \
    --model_name Isomag2D_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams True \
    --device cuda:0

python Code/octree_upscaling.py \
    --octree_file Isomag2D_4xreduction.octree \
    --volume_file Isomag2D.h5 \
    --save_name Isomag2D_4xreduction_HSR_seams \
    --upscaling_method model \
    --model_name Isomag2D_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams True \
    --device cuda:0

################################

python Code/octree_upscaling.py \
    --octree_file Vorts_8xreduction.octree \
    --volume_file Vorts.h5 \
    --save_name Vorts_8xreduction_HSR_noseams \
    --upscaling_method model \
    --model_name vorts_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams False \
    --device cuda:0

python Code/octree_upscaling.py \
    --octree_file Vorts_8xreduction.octree \
    --volume_file Vorts.h5 \
    --save_name Vorts_8xreduction_nearest \
    --upscaling_method nearest \
    --model_name vorts_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams True \
    --device cuda:0

python Code/octree_upscaling.py \
    --octree_file Vorts_8xreduction.octree \
    --volume_file Vorts.h5 \
    --save_name Vorts_8xreduction_HSR_seams \
    --upscaling_method model \
    --model_name vorts_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams True \
    --device cuda:0

#################################

python Code/octree_upscaling.py \
    --octree_file Nyx.octree \
    --volume_file Nyx.h5 \
    --save_name Nyx_HSR \
    --upscaling_method model \
    --model_name nyx256_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams False \
    --device cuda:0

python Code/octree_upscaling.py \
    --octree_file Nyx.octree \
    --volume_file Nyx.h5 \
    --save_name Nyx_HSR_seams \
    --upscaling_method model \
    --model_name nyx256_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams True \
    --device cuda:0

#################################

python Code/octree_upscaling.py \
    --octree_file Mixing.octree \
    --volume_file Mixing3D.h5 \
    --save_name Mixing_HSR \
    --upscaling_method model \
    --model_name mixing3D_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams False \
    --device cuda:0

python Code/octree_upscaling.py \
    --octree_file Mixing.octree \
    --volume_file Mixing3D.h5 \
    --save_name Mixing_HSR_seams \
    --upscaling_method model \
    --model_name mixing3D_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams True \
    --device cuda:0

#################################

python Code/octree_upscaling.py \
    --octree_file HeatedCylinder_16x_v2.octree \
    --volume_file HeatedCylinder.h5 \
    --save_name HeatedCylinder_HSR \
    --upscaling_method model \
    --model_name boussinesq_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams False \
    --device cuda:0

python Code/octree_upscaling.py \
    --octree_file HeatedCylinder_16x_v2.octree \
    --volume_file HeatedCylinder.h5 \
    --save_name HeatedCylinder_HSR_seams \
    --upscaling_method model \
    --model_name boussinesq_ESRGAN_new \
    --save_original_volume True \
    --save_error_volume False \
    --seams True \
    --device cuda:0