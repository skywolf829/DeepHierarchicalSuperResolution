import os
import json

class Options():
    def get_default():
        opt = {}
        # Input info
        opt["mode"]                    = "3D"      # 2D or 3D
        opt['model']                   = "ESRGAN"  # ESRGAN or SSRTVD 
        opt["data_folder"]             = "Isomag3D"
        opt["save_folder"]             = "SavedModels"
        opt["save_name"]               = "Temp"    # Folder that the model will be saved to
        opt["num_channels"]            = 1
        opt["min_dimension_size"]      = 16        # Smallest a dimension can go to upscale from
        opt["cropping_resolution"]     = 96
        opt["train_date_time"]         = None      # The day/time the model was trained (finish time)
        opt['scale_factor']            = 4         # For SSRTVD, the scale factor to increase input by

        opt['random_flipping']         = True
        opt["num_workers"]             = 2

        # generator info
        opt["num_blocks"]              = 3
        opt['num_discrim_blocks']      = 5
        opt["num_kernels"]             = 96        # Num of kernels in smallest scale conv layers
        opt["kernel_size"]             = 3
        opt["padding"]                 = 1
        opt["stride"]                  = 1
        opt['B']                      = 0.2

        opt["n"]                       = 0         # Number of scales in the heirarchy, defined by the input and min_dimension_size
        opt["resolutions"]             = []        # The scales for the GAN

        opt["train_distributed"]       = False
        opt["device"]                  = "cuda:0"
        opt["gpus_per_node"]           = 8
        opt["num_nodes"]               = 1
        opt["ranking"]                 = 0

        opt["save_generators"]         = True
        opt["save_discriminators"]     = True
        opt["patch_size"]              = 96
        opt["training_patch_size"]     = 96

        # GAN training info
        opt["alpha_1"]                 = 1       # Reconstruction loss coefficient
        opt["alpha_2"]                 = 0.1        # Adversarial loss coefficient

        opt["generator_steps"]         = 1
        opt["discriminator_steps"]     = 1
        opt["epochs"]                  = 50
        opt["minibatch"]               = 1        # Minibatch for training
        opt["g_lr"]                    = 0.0001    # Learning rate for GAN generator
        opt["d_lr"]                    = 0.0004    # Learning rate for GAN discriminator
        opt["beta_1"]                  = 0.5
        opt["beta_2"]                  = 0.999
        opt["gamma"]                   = 0.1

        # Info during training (to continue if it stopped)
        opt["scale_in_training"]       = 0
        opt["iteration_number"]        = 0
        opt["epoch_number"]            = 0
        opt["save_every"]              = 100
        opt["save_training_loss"]      = True

        return opt

def save_options(opt, save_location):
    with open(os.path.join(save_location, "options.json"), 'w') as fp:
        json.dump(opt, fp, sort_keys=True, indent=4)
    
def load_options(load_location):
    opt = Options.get_default()
    print(load_location)
    if not os.path.exists(load_location):
        print("%s doesn't exist, load failed" % load_location)
        return
        
    if os.path.exists(os.path.join(load_location, "options.json")):
        with open(os.path.join(load_location, "options.json"), 'r') as fp:
            opt2 = json.load(fp)
    else:
        print("%s doesn't exist, load failed" % "options.json")
        return
    
    # For forward compatibility with new attributes in the options file
    for attr in opt2.keys():
        opt[attr] = opt2[attr]

    return opt
