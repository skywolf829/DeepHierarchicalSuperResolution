#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/TestingScripts/test_SSR.py \
--mode 3D --data_folder Supernova \
--model_name supernova_ESRGAN_new \
--dict_entry_name ESRGAN \
--device cuda:0 \
--parallel True \
--test_on_gpu True \
--output_file_name Supernova.results

python -u Code/TestingScripts/test_SSR.py \
--mode 3D --data_folder Supernova \
--model_name supernova_SSRTVD_NO_D_new \
--dict_entry_name SSRTVD \
--device cuda:0 \
--parallel True \
--test_on_gpu True \
--output_file_name Supernova.results

python -u Code/TestingScripts/test_SSR.py \
--mode 3D --data_folder Supernova \
--model_name supernova_STNet_new \
--dict_entry_name STNet \
--device cuda:0 \
--parallel True \
--test_on_gpu True \
--output_file_name Supernova.results