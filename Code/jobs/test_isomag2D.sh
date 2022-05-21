#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/TestingScripts/test_SSR.py \
--mode 2D --data_folder Isomag2D \
--model_name isomag2D_ESRGAN_new \
--dict_entry_name ESRGAN \
--device cuda:0 \
--parallel False \
--test_on_gpu True \
--output_file_name Isomag2D.results

python -u Code/TestingScripts/test_SSR.py \
--mode 2D --data_folder Isomag2D \
--model_name isomag2D_SSRTVD_NO_D_new \
--dict_entry_name SSRTVD \
--device cuda:0 \
--parallel False \
--test_on_gpu True \
--output_file_name Isomag2D.results

python -u Code/TestingScripts/test_SSR.py \
--mode 2D --data_folder Isomag2D \
--model_name isomag2D_STNet_new \
--dict_entry_name STNet \
--device cuda:0 \
--parallel False \
--test_on_gpu True \
--output_file_name Isomag2D.results