#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/DeepHierarchicalSuperResolution

python -u Code/VisualizationScripts/vis_SR_results.py \
--save_folder isomag2D_vis_results \
--output_file_name Isomag2D.results \
--mode 2D \
--start_ts 4000 \
--ts_skip 10

python -u Code/VisualizationScripts/vis_SR_results.py \
--save_folder isomag3D_vis_results \
--output_file_name Isomag3D.results \
--mode 3D \
--start_ts 4000 \
--ts_skip 100

python -u Code/VisualizationScripts/vis_SR_results.py \
--save_folder plume_vis_results \
--output_file_name Plume.results \
--mode 3D \
--start_ts 21 \
--ts_skip 1

python -u Code/VisualizationScripts/vis_SR_results.py \
--save_folder vorts_vis_results \
--output_file_name Vorts.results \
--mode 3D \
--start_ts 21 \
--ts_skip 1

python -u Code/VisualizationScripts/vis_SR_results.py \
--save_folder nyx256_vis_results \
--output_file_name Nyx256.results \
--mode 3D \
--start_ts 0 \
--ts_skip 1

python -u Code/VisualizationScripts/vis_SR_results.py \
--save_folder supernova_vis_results \
--output_file_name Supernova.results \
--mode 3D \
--start_ts 1335 \
--ts_skip 1