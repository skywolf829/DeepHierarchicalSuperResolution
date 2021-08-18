#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python3 -u Code/TestingScripts/test_SSR.py \
--mode 3D --data_folder Isomag3D --model_name Isomag3D \
--device cuda:0 --parallel True --test_on_gpu True \
--output_file_name Isomag3D.results