import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utility_functions import load_obj, save_obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')
    
    parser.add_argument('--save_folder',default="Isomag2D_datareduction",
        type=str,help='Folder to save images to')
    parser.add_argument('--min_PSNR',default=30.0, type=float)    
    parser.add_argument('--max_PSNR',default=60.0, type=float)    
    
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..", "..")
    data_folder = os.path.join(project_folder_path, "Data", "DataReduction")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(output_folder, args['save_folder'])
    results_file = os.path.join(save_folder, "results.pkl")
    
    make_all = False
    full_file_size = 4096
    #full_file_size = 4194306
    #full_file_size = 524209
    #full_file_size = 8192
    #full_file_size = 32770    

    results = load_obj(results_file)
    #save_obj(results, results_file)
       


    font = {#'font.family' : 'normal',
        #'font.weight' : 'bold',
        'font.size'   : 12}
    plt.rcParams.update(font)

    for method in results.keys():
        file_size = results[method]['file_size']
        compression_ratios = full_file_size / np.array(file_size)
        results[method]['compression_ratio'] = compression_ratios


    for method in results.keys():
        print(results[method].keys())
        plt.figure(1)
        y_psnr = np.array(results[method]['rec_psnr'])
        y_ssim = np.array(results[method]['rec_ssim'])
        x_cr = np.array(results[method]["compression_ratio"])
        low = np.where(y_psnr > args['min_PSNR'], y_psnr, 0)
        high = np.where(y_psnr < args['max_PSNR'], y_psnr, 0)
        in_range = np.argwhere(low*high).flatten()
        print(in_range)
        y_psnr = y_psnr[in_range]
        y_ssim = y_ssim[in_range]
        x = x_cr[in_range]
        psnr_ordering = np.argsort(y_psnr)     
        ssim_ordering = np.argsort(y_ssim)     

        plt.plot(x[psnr_ordering], y_psnr[psnr_ordering], label=method)
        plt.figure(2)
        plt.plot(x[ssim_ordering], y_ssim[ssim_ordering], label=method)
        plt.figure(3)
        plt.plot(np.array(results[method]['psnrs']), np.array(results[method]['rec_psnr']), label=method)
        
    plt.figure(1)
    plt.ylabel("Reconstructed data PSNR (dB)")
    plt.xlabel("Compression ratio")
    plt.title("Reconstructed data PSNR over compression ratios")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "psnr.png"))
    #plt.show()
    plt.clf()

    plt.figure(2)
    plt.ylabel("Reconstructed data SSIM")
    plt.xlabel("Compression ratio")
    plt.title("Reconstructed data SSIM over compression ratios")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "ssim.png"))
    #plt.show()
    plt.clf()

    plt.figure(3)
    plt.ylabel("Reconstructed PSNR (dB)")
    plt.xlabel("Target PSNR")
    plt.title("Reconstructed PSNR over target PSNR")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "psnr_vs_recpsnr.png"))
    #plt.show()
    plt.clf()