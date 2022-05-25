#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/TestingScripts/test_SSR.py \
--mode 2D --data_folder boussinesq \
--model_name boussinesq_ESRGAN_new \
--dict_entry_name ESRGAN \
--device cuda:0 \
--parallel False \
--test_on_gpu True \
--output_file_name Boussinesq.results

python -u Code/TestingScripts/test_SSR.py \
--mode 2D --data_folder boussinesq \
--model_name boussinesq_SSRTVD_NO_D_new \
--dict_entry_name SSRTVD \
--device cuda:0 \
--parallel False \
--test_on_gpu True \
--output_file_name Boussinesq.results

python -u Code/TestingScripts/test_SSR.py \
--mode 2D --data_folder boussinesq \
--model_name boussinesq_STNet_new \
--dict_entry_name STNet \
--device cuda:0 \
--parallel False \
--test_on_gpu True \
--output_file_name Boussinesq.results