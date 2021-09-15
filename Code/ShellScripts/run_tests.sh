#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

#python3 -u Code/TestingScripts/test_SSR.py \
#--mode 2D --data_folder Isomag2D --model_name Isomag2D \
#--device cuda:0 --parallel False --test_on_gpu True \
#--output_file_name Isomag2D.results --dict_entry_name model

#python3 -u Code/TestingScripts/test_SSR.py \
#--mode 2D --data_folder Isomag2D --model_name Isomag2D_noGAN \
#--device cuda:0 --parallel False --test_on_gpu True \
#--output_file_name Isomag2D.results --dict_entry_name model_noGAN

#python3 -u Code/TestingScripts/test_SSR.py \
#--mode 3D --data_folder Isomag3D --model_name Isomag3D \
#--device cuda:0 --parallel True --test_on_gpu True \
#--output_file_name Isomag3D.results --dict_entry_name model

#python3 -u Code/TestingScripts/test_SSR.py \
#--mode 3D --data_folder Isomag3D --model_name Isomag3D_noGAN \
#--device cuda:0 --parallel True --test_on_gpu True \
#--output_file_name Isomag3D.results --dict_entry_name model_noGAN

python3 -u Code/TestingScripts/test_SSR.py \
--mode 3D --data_folder Mixing3D --model_name Mixing3D \
--device cuda:0 --parallel True --test_on_gpu True \
--output_file_name Mixing3D.results --dict_entry_name model

python3 -u Code/TestingScripts/test_SSR.py \
--mode 3D --data_folder Mixing3D --model_name Mixing3D_noGAN \
--device cuda:0 --parallel True --test_on_gpu True \
--output_file_name Mixing3D.results --dict_entry_name model_noGAN

#python3 -u Code/TestingScripts/test_SSR.py \
#--mode 3D --data_folder Plume --model_name Plume \
#--device cuda:0 --parallel False --test_on_gpu True \
#--output_file_name Plume.results --dict_entry_name model

#python3 -u Code/TestingScripts/test_SSR.py \
#--mode 3D --data_folder Plume --model_name Plume_noGAN \
#--device cuda:0 --parallel False --test_on_gpu True \
#--output_file_name Plume.results --dict_entry_name model_noGAN

#python3 -u Code/TestingScripts/test_SSR.py \
#--mode 3D --data_folder Vorts --model_name Vorts \
#--device cuda:0 --parallel False --test_on_gpu True \
#--output_file_name Vorts.results --dict_entry_name model

#python3 -u Code/TestingScripts/test_SSR.py \
#--mode 3D --data_folder Vorts --model_name Vorts_noGAN \
#--device cuda:0 --parallel False --test_on_gpu True \
#--output_file_name Vorts.results --dict_entry_name model_noGAN


