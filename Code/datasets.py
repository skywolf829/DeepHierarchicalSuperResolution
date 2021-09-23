import os
import torch
import h5py
from utility_functions import AvgPool3D, AvgPool2D

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data", "SuperResolutionData")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")


class SSRTVD_dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.items = []
        self.item_names = []
        self.subsample_dist = 1

        folder_to_load = os.path.join(data_folder, self.opt['data_folder'], "TrainingData")

        print("Initializing dataset - reading %i items" % len(os.listdir(folder_to_load)))
        for filename in os.listdir(folder_to_load):
            self.item_names.append(filename)            
            
            print("Loading " + filename)   
            f = h5py.File(os.path.join(folder_to_load, filename), 'r')
            d = torch.tensor(f.get('data'))
            f.close()
            self.items.append(d)
        self.resolution = self.items[0].shape
        print("Resolution: " + str(self.resolution))

    def __len__(self):
        return len(self.items)-3

    def resolution(self):
        return self.resolution

    def get_patch_ranges(self, frame, patch_size, receptive_field, mode):
        starts = []
        rf = receptive_field
        ends = []
        if(mode == "3D"):
            for z in range(0,max(1,frame.shape[2]), patch_size-2*rf):
                z = min(z, max(0, frame.shape[2] - patch_size))
                z_stop = min(frame.shape[2], z + patch_size)
                
                for y in range(0, max(1,frame.shape[3]), patch_size-2*rf):
                    y = min(y, max(0, frame.shape[3] - patch_size))
                    y_stop = min(frame.shape[3], y + patch_size)

                    for x in range(0, max(1,frame.shape[4]), patch_size-2*rf):
                        x = min(x, max(0, frame.shape[4] - patch_size))
                        x_stop = min(frame.shape[4], x + patch_size)

                        starts.append([z, y, x])
                        ends.append([z_stop, y_stop, x_stop])
        elif(mode == "2D" or mode == "3Dto2D"):
            for y in range(0, max(1,frame.shape[2]-patch_size+1), patch_size-2*rf):
                y = min(y, max(0, frame.shape[2] - patch_size))
                y_stop = min(frame.shape[2], y + patch_size)

                for x in range(0, max(1,frame.shape[3]-patch_size+1), patch_size-2*rf):
                    x = min(x, max(0, frame.shape[3] - patch_size))
                    x_stop = min(frame.shape[3], x + patch_size)

                    starts.append([y, x])
                    ends.append([y_stop, x_stop])
        return starts, ends

    def set_subsample_dist(self,dist):
        self.subsample_dist = dist
        
    def __getitem__(self, index):
        data = self.items[index:index+3]
        data = torch.cat(data, dim=0).clone()

        x_start = 0
        x_end = data.shape[1]
        y_start = 0
        y_end = data.shape[2]

        if(self.opt['mode'] == "3D"):
            z_start = 0
            z_end = data.shape[3]
            if((z_end-z_start) / self.subsample_dist > self.opt['cropping_resolution']):
                z_start = torch.randint(data.shape[3] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                z_end = z_start + self.opt['cropping_resolution']*self.subsample_dist

        if((y_end-y_start) / self.subsample_dist > self.opt['cropping_resolution']):
            y_start = torch.randint(data.shape[2] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
            y_end = y_start + self.opt['cropping_resolution']*self.subsample_dist
        if((x_end-x_start) / self.subsample_dist > self.opt['cropping_resolution']):
            x_start = torch.randint(data.shape[1] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
            x_end = x_start + self.opt['cropping_resolution']*self.subsample_dist
        
        
        if(self.opt['mode'] == "3D"):
            data = data[:,x_start:x_end,
                y_start:y_end,
                z_start:z_end]
        elif(self.opt['mode'] == "2D"):
            data =  data[:,x_start:x_end,
                y_start:y_end]
        
        if(self.subsample_dist > 1):
            if(self.opt["mode"] == "3D"):
                data = AvgPool3D(data.unsqueeze(0), self.subsample_dist)[0]
            elif(self.opt['mode'] == "2D"):
                data = AvgPool2D(data.unsqueeze(0), self.subsample_dist)[0]
                
        
        if(self.opt['random_flipping']):
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[1])
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[2])
            if(self.opt['mode'] == "3D"):
                if(torch.rand(1).item() > 0.5):
                    data = torch.flip(data,[3])

        return data

class TestingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name):

        folder_to_load = os.path.join(data_folder, dataset_name, "TestingData")
        print("Initializing dataset")
        self.item_names = []
        for filename in os.listdir(folder_to_load):
            self.item_names.append(filename.split("/")[-1].split(".")[0])
            self.ext = filename.split("/")[-1].split(".")[1]
        self.item_names.sort(key=int)
        print("Dataset has " + str(len(self.item_names)) + " items. Reading them now.")

        self.items = []

        for filename in self.item_names:
            to_load = os.path.join(folder_to_load, filename + "." + self.ext)
            
            print("Loading " + filename)   
            f = h5py.File(to_load, 'r')
            d = torch.tensor(f.get('data'))
            f.close()
            self.items.append(d)    

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):       
        return self.items[index]

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.items = []
        self.item_names = []
        self.subsample_dist = 1

        folder_to_load = os.path.join(data_folder, self.opt['data_folder'], "TrainingData")

        print("Initializing dataset - reading %i items" % len(os.listdir(folder_to_load)))
        for filename in os.listdir(folder_to_load):
            self.item_names.append(filename)            
            
            print("Loading " + filename)   
            f = h5py.File(os.path.join(folder_to_load, filename), 'r')
            d = torch.tensor(f.get('data'))
            f.close()
            self.items.append(d)
        self.resolution = self.items[0].shape
        print("Resolution: " + str(self.resolution))

    def __len__(self):
        return len(self.items)

    def resolution(self):
        return self.resolution

    def get_patch_ranges(self, frame, patch_size, receptive_field, mode):
        starts = []
        rf = receptive_field
        ends = []
        if(mode == "3D"):
            for z in range(0,max(1,frame.shape[2]), patch_size-2*rf):
                z = min(z, max(0, frame.shape[2] - patch_size))
                z_stop = min(frame.shape[2], z + patch_size)
                
                for y in range(0, max(1,frame.shape[3]), patch_size-2*rf):
                    y = min(y, max(0, frame.shape[3] - patch_size))
                    y_stop = min(frame.shape[3], y + patch_size)

                    for x in range(0, max(1,frame.shape[4]), patch_size-2*rf):
                        x = min(x, max(0, frame.shape[4] - patch_size))
                        x_stop = min(frame.shape[4], x + patch_size)

                        starts.append([z, y, x])
                        ends.append([z_stop, y_stop, x_stop])
        elif(mode == "2D" or mode == "3Dto2D"):
            for y in range(0, max(1,frame.shape[2]-patch_size+1), patch_size-2*rf):
                y = min(y, max(0, frame.shape[2] - patch_size))
                y_stop = min(frame.shape[2], y + patch_size)

                for x in range(0, max(1,frame.shape[3]-patch_size+1), patch_size-2*rf):
                    x = min(x, max(0, frame.shape[3] - patch_size))
                    x_stop = min(frame.shape[3], x + patch_size)

                    starts.append([y, x])
                    ends.append([y_stop, x_stop])
        return starts, ends

    def set_subsample_dist(self,dist):
        self.subsample_dist = dist
        
    def __getitem__(self, index):
        data = self.items[index]
 
        x_start = 0
        x_end = data.shape[1]
        y_start = 0
        y_end = data.shape[2]

        if(self.opt['mode'] == "3D"):
            z_start = 0
            z_end = data.shape[3]
            if((z_end-z_start) / self.subsample_dist > self.opt['cropping_resolution']):
                z_start = torch.randint(data.shape[3] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                z_end = z_start + self.opt['cropping_resolution']*self.subsample_dist

        if((y_end-y_start) / self.subsample_dist > self.opt['cropping_resolution']):
            y_start = torch.randint(data.shape[2] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
            y_end = y_start + self.opt['cropping_resolution']*self.subsample_dist
        if((x_end-x_start) / self.subsample_dist > self.opt['cropping_resolution']):
            x_start = torch.randint(data.shape[1] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
            x_end = x_start + self.opt['cropping_resolution']*self.subsample_dist
        
        
        if(self.opt['mode'] == "3D"):
            data = data[:,x_start:x_end,
                y_start:y_end,
                z_start:z_end]
        elif(self.opt['mode'] == "2D"):
            data =  data[:,x_start:x_end,
                y_start:y_end]
        
        if(self.subsample_dist > 1):
            if(self.opt["mode"] == "3D"):
                data = AvgPool3D(data.unsqueeze(0), self.subsample_dist)[0]
            elif(self.opt['mode'] == "2D"):
                data = AvgPool2D(data.unsqueeze(0), self.subsample_dist)[0]
                
        
        if(self.opt['random_flipping']):
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[1])
            if(torch.rand(1).item() > 0.5):
                data = torch.flip(data,[2])
            if(self.opt['mode'] == "3D"):
                if(torch.rand(1).item() > 0.5):
                    data = torch.flip(data,[3])

        return data.clone()


def normalize_folders():
    folders = ['Mixing2D', 'Vorts', 'Plume', 'Mixing3D', 'Isomag2D', 'Isomag3D']
    subfolders = ['TrainingData', 'TestingData']

    for folder in folders:
        for subfolder in subfolders:
            for f_name in os.listdir(os.path.join(data_folder, folder, subfolder)):
                print(os.path.join(folder, subfolder, f_name))
                f = h5py.File(os.path.join(data_folder, folder, subfolder, f_name), 'r+')
                f_data = torch.tensor(f['data'])
                f_data -= f_data.min()
                f_data /= (f_data.max() + 1e-6)
                del f['data']
                f['data'] = f_data.numpy()
                f.close()

                f = h5py.File(os.path.join(data_folder, folder, subfolder, f_name), 'r')
                f_data = torch.tensor(f['data'])
                print("Min: %0.04f, max %0.04f, avg %0.04f" % (f_data.min(), f_data.max(), f_data.mean()))
                f.close()
            print("Finished " + os.path.join(folder, subfolder))