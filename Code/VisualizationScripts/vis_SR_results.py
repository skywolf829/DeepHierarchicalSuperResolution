import numpy as np
import matplotlib.pyplot as plt
import argparse

from numpy.lib.npyio import save
from utility_functions import load_obj
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')
    
    parser.add_argument('--save_folder',default="Isomag3D_vis_results",
        type=str,help='Folder to save images to')
    parser.add_argument('--output_file_name',default="Isomag3D.results",
        type=str,help='filename to visualize in output folder')
    parser.add_argument('--mode',type=str,default="3D")
    parser.add_argument('--start_ts', default=4000, type=int)
    parser.add_argument('--ts_skip', default=100, type=int)
    
    font = {#'font.family' : 'normal',
        #'font.weight' : 'bold',
        'font.size'   : 15}
    plt.rcParams.update(font)

    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..", "..")
    data_folder = os.path.join(project_folder_path, "Data", "SuperResolutionData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    results_file = os.path.join(output_folder, args['output_file_name'])
    
    results = load_obj(results_file)
    save_folder = os.path.join(output_folder, args['save_folder'])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for scale_factor in results.keys():
        if not os.path.exists(os.path.join(save_folder, scale_factor)):
            os.makedirs(os.path.join(save_folder, scale_factor))
    

    
    interp = "bilinear" if args['mode'] == "2D" else "trilinear"

    for scale_factor in results.keys():
        model_results = results[scale_factor]["model"]
        interp_results = results[scale_factor][interp]


        for metric in model_results.keys():
            fig = plt.figure()
            y_label = metric

            # model results plotting
            x = np.arange(args['start_ts'], 
                args['start_ts'] + args['ts_skip']*len(model_results[metric]),
                args['ts_skip'])
            y = model_results[metric]
            plt.plot(x, y, label="model")

            # interpolation results plotting
            x = np.arange(args['start_ts'], 
                args['start_ts'] + args['ts_skip']*len(interp_results[metric]),
                args['ts_skip'])
            y = interp_results[metric]
            plt.plot(x, y, label=interp)

            plt.legend()
            plt.xlabel("Timestep")
            plt.ylabel(y_label)

            plt.title(scale_factor + " SR - " + metric)
            plt.savefig(os.path.join(save_folder, scale_factor, metric+".png"))
            plt.clf()
