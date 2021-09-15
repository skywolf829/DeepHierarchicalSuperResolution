import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import matplotlib.pyplot as plt
import argparse
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
        'font.size'   : 13}
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
    if not os.path.exists(os.path.join(save_folder, "MedianValues")):
        os.makedirs(os.path.join(save_folder, "MedianValues"))
    

    
    interp = "bilinear" if args['mode'] == "2D" else "trilinear"

    for scale_factor in results.keys():
        print(scale_factor)
        model_results = results[scale_factor]["model"]
        model_noGAN_results = results[scale_factor]["model_noGAN"]
        interp_results = results[scale_factor][interp]


        for metric in model_results.keys():
            fig = plt.figure()
            y_label = metric

            # model results plotting
            x = np.arange(args['start_ts'], 
                args['start_ts'] + args['ts_skip']*len(model_results[metric]),
                args['ts_skip'])
            y = model_results[metric]
            plt.plot(x, y, label="GAN")

            # model_noGAN results plotting
            x = np.arange(args['start_ts'], 
                args['start_ts'] + args['ts_skip']*len(model_noGAN_results[metric]),
                args['ts_skip'])
            y = model_noGAN_results[metric]
            plt.plot(x, y, label="CNN")

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

    # Overall graphs

    averaged_results = {}
    averaged_results['model'] = {}
    averaged_results['model_noGAN'] = {}
    averaged_results[interp] = {}

    scale_factors = []

    for scale_factor in results.keys():
        model_results = results[scale_factor]["model"]
        model_noGAN_results = results[scale_factor]["model_noGAN"]
        interp_results = results[scale_factor][interp]

        scale_factor_int = int(scale_factor.split('x')[0])
        scale_factors.append(scale_factor_int)

        for metric in model_results.keys():
            if(metric not in averaged_results['model'].keys()):
                averaged_results['model'][metric] = []
                averaged_results['model_noGAN'][metric] = []
                averaged_results[interp][metric] = []
            averaged_results['model'][metric].append(np.median(np.array(model_results[metric])))
            averaged_results['model_noGAN'][metric].append(np.median(np.array(model_noGAN_results[metric])))
            averaged_results[interp][metric].append(np.median(np.array(interp_results[metric])))

    
    for metric in model_results.keys():
        fig = plt.figure()
        y_label = metric

        # model results plotting
        x = scale_factors
        y = averaged_results['model'][metric]
        plt.plot(x, y, label="GAN")

        # model_noGAN results plotting
        x = scale_factors
        y = averaged_results['model_noGAN'][metric]
        plt.plot(x, y, label="CNN")

        # interpolation results plotting
        x = scale_factors
        y = averaged_results[interp][metric]
        plt.plot(x, y, label=interp)

        plt.legend()
        plt.xlabel("Scale factor")
        plt.ylabel(y_label)
        plt.xscale('log')
        plt.minorticks_off()
        plt.xticks(scale_factors, labels=scale_factors)
        plt.title("Median " + metric + " over SR factors")
        plt.savefig(os.path.join(save_folder, "MedianValues", metric+".png"))
        plt.clf()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    left_y_label = "PSNR (dB)"
    right_y_label = "SSIM"

    # model results plotting
    x = scale_factors
    left_y = averaged_results['model'][left_y_label]
    right_y = averaged_results['model'][right_y_label]
    ax1.plot(x, left_y, label='GAN', marker='s')
    ax2.plot(x, right_y, label='GAN', linestyle='dashed', marker='^')

    # model noGAN
    x = scale_factors
    left_y = averaged_results['model_noGAN'][left_y_label]
    right_y = averaged_results['model_noGAN'][right_y_label]
    ax1.plot(x, left_y, label='CNN', marker='s')
    ax2.plot(x, right_y, label='CNN', linestyle='dashed', marker='^')

    # interpolation results plotting
    x = scale_factors
    left_y = averaged_results[interp][left_y_label]
    right_y = averaged_results[interp][right_y_label]
    ax1.plot(x, left_y, label=interp, marker='s')
    ax2.plot(x, right_y, label=interp, linestyle='dashed', marker='^')

    ax1.legend()
    #ax2.legend()
    ax1.set_xlabel("Scale factor")
    ax1.set_ylabel(left_y_label)
    ax2.set_ylabel(right_y_label)

    ax1.set_xscale('log')
    ax1.minorticks_off()

    ax1.set_xticks(scale_factors)
    ax1.set_xticklabels(scale_factors)
    ax1.set_title("Median PSNR/SSIM over SR factors")
    plt.savefig(os.path.join(save_folder, "MedianValues", "Combined.png"))
    plt.clf()