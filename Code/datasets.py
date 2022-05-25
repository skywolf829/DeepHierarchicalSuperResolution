import os
import torch
import h5py
from utility_functions import AvgPool3D, AvgPool2D
import numpy as np


project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data", "SuperResolutionData")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")


class SSRTVD_dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.items = []
        self.subsample_dist = 1

        folder_to_load = os.path.join(data_folder, 
                                      self.opt['data_folder'], 
                                      "TrainingData")

        print("Initializing dataset - reading %i items" % len(os.listdir(folder_to_load)))
        filenames = []
        filenames_ints = []
        for filename in os.listdir(folder_to_load):   
            filenames.append(filename)
            filenames_ints.append(int(filename.split(".")[0]))
        
        sorted_order = np.argsort(np.array(filenames_ints))
        for i in range(len(sorted_order)):
            filename = filenames[sorted_order[i]]
            print("Loading " + filename)   
            f = h5py.File(os.path.join(folder_to_load, filename), 'r')
            d = torch.tensor(np.array(f.get('data')))
            f.close()
            self.items.append(d)
        self.resolution = self.items[0].shape
        print("Resolution: " + str(self.resolution))

    def __len__(self):
        return len(self.items)-2

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

        folder_to_load = os.path.join(data_folder, 
                                      dataset_name, 
                                      "TestingData")
        print("Initializing dataset")
        filenames = []
        filenames_ints = []
        for filename in os.listdir(folder_to_load):   
            filenames.append(filename)
            filenames_ints.append(int(filename.split(".")[0]))
        
        sorted_order = np.argsort(np.array(filenames_ints))
        print("Dataset has " + str(len(filenames)) + " items. Reading them now.")
        
        self.items = []

        for i in range(len(sorted_order)):
            filename = filenames[sorted_order[i]]
            to_load = os.path.join(folder_to_load, filename)
            
            print("Loading " + filename)   
            f = h5py.File(to_load, 'r')
            d = torch.tensor(np.array(f.get('data'))).type(torch.float32)
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
        filenames = []
        filenames_ints = []
        for filename in os.listdir(folder_to_load):   
            filenames.append(filename)
            filenames_ints.append(int(filename.split(".")[0]))
        
        sorted_order = np.argsort(np.array(filenames_ints))
        print("Dataset has " + str(len(self.item_names)) + " items. Reading them now.")
        
        self.items = []

        for i in range(len(sorted_order)):
            filename = filenames[sorted_order[i]]
            self.item_names.append(filename)            
            
            print("Loading " + filename)   
            f = h5py.File(os.path.join(folder_to_load, filename), 'r')
            d = np.array(f.get('data'))
            d = torch.tensor(d).type(torch.float32)
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

class NyxUseCaseTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.items = []
        self.item_names = []
        self.subsample_dist = 1

        folder_to_load_LR = os.path.join(data_folder, self.opt['data_folder'], "TrainingDataLR")
        folder_to_load_HR = os.path.join(data_folder, self.opt['data_folder'], "TrainingDataHR")
        
        filenames = []
        filenames_ints = []
        for filename in os.listdir(folder_to_load_LR):   
            filenames.append(filename)
            filenames_ints.append(int(filename.split(".")[0]))
        
        sorted_order = np.argsort(np.array(filenames_ints))
        print("Dataset has " + str(len(self.item_names)) + " items. Reading them now.")
        
        self.items_lr = []
        self.items_hr = []

        for i in range(len(sorted_order)):
            filename = filenames[sorted_order[i]]
            self.item_names.append(filename)            
            
            print("Loading " + filename)   
            f = h5py.File(os.path.join(folder_to_load_LR, filename), 'r')
            d = np.array(f.get('data'))
            d = torch.tensor(d).type(torch.float32)
            f.close()
            self.items_lr.append(d)
            
            f = h5py.File(os.path.join(folder_to_load_HR, filename), 'r')
            d = np.array(f.get('data'))
            d = torch.tensor(d).type(torch.float32)
            f.close()
            self.items_hr.append(d)
            
        self.resolution = self.items_hr[0].shape
        print("Resolution: " + str(self.resolution))

    def __len__(self):
        return len(self.items_hr)

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
        
    def get_item_with_LR(self, index):
        data_LR = self.items_lr[index].clone()
        data_HR = self.items_hr[index].clone()
        data_HR = AvgPool3D(data_HR.unsqueeze(0), 2)[0]
        
        x_start = 0
        x_end = data_LR.shape[1]
        y_start = 0
        y_end = data_LR.shape[2]

        if(self.opt['mode'] == "3D"):
            z_start = 0
            z_end = data_LR.shape[3]
            if((z_end-z_start)*2 > self.opt['cropping_resolution']):
                z_start = torch.randint(data_LR.shape[3] - int(self.opt['cropping_resolution']/2), [1]).item()
                z_end = z_start + int(self.opt['cropping_resolution']/2)

        if((y_end-y_start)*2 > self.opt['cropping_resolution']):
            y_start = torch.randint(data_LR.shape[2] - int(self.opt['cropping_resolution']/2), [1]).item()
            y_end = y_start + int(self.opt['cropping_resolution']/2)
        if((x_end-x_start)*2 > self.opt['cropping_resolution']):
            x_start = torch.randint(data_LR.shape[1] - int(self.opt['cropping_resolution']/2), [1]).item()
            x_end = x_start + int(self.opt['cropping_resolution']/2)
        
        
        data_HR = data_HR[:,x_start*2:x_end*2,
            y_start*2:y_end*2,
            z_start*2:z_end*2]
        data_LR = data_LR[:,x_start:x_end,
            y_start:y_end,
            z_start:z_end]
        
        if(self.opt['random_flipping']):
            if(torch.rand(1).item() > 0.5):
                data_HR = torch.flip(data_HR,[1])
                data_LR = torch.flip(data_LR,[1])
            if(torch.rand(1).item() > 0.5):
                data_HR = torch.flip(data_HR,[2])
                data_LR = torch.flip(data_LR,[2])
            if(self.opt['mode'] == "3D"):
                if(torch.rand(1).item() > 0.5):
                    data_HR = torch.flip(data_HR,[3])
                    data_LR = torch.flip(data_LR,[3])
        
        return data_HR, data_LR
        
    def __getitem__(self, index):
        if(self.subsample_dist == 2):
            return self.get_item_with_LR(index)
        data_HR = self.items_hr[index].clone()
 
        x_start = 0
        x_end = data_HR.shape[1]
        y_start = 0
        y_end = data_HR.shape[2]

        if(self.opt['mode'] == "3D"):
            z_start = 0
            z_end = data_HR.shape[3]
            if((z_end-z_start) / self.subsample_dist > self.opt['cropping_resolution']):
                z_start = torch.randint(data_HR.shape[3] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
                z_end = z_start + self.opt['cropping_resolution']*self.subsample_dist

        if((y_end-y_start) / self.subsample_dist > self.opt['cropping_resolution']):
            y_start = torch.randint(data_HR.shape[2] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
            y_end = y_start + self.opt['cropping_resolution']*self.subsample_dist
        if((x_end-x_start) / self.subsample_dist > self.opt['cropping_resolution']):
            x_start = torch.randint(data_HR.shape[1] - self.opt['cropping_resolution']*self.subsample_dist, [1]).item()
            x_end = x_start + self.opt['cropping_resolution']*self.subsample_dist
        
        
        if(self.opt['mode'] == "3D"):
            data_HR = data_HR[:,x_start:x_end,
                y_start:y_end,
                z_start:z_end]
        elif(self.opt['mode'] == "2D"):
            data_HR =  data_HR[:,x_start:x_end,
                y_start:y_end]
        
        if(self.subsample_dist > 1):
            if(self.opt["mode"] == "3D"):
                data_HR = AvgPool3D(data_HR.unsqueeze(0), self.subsample_dist)[0]
            elif(self.opt['mode'] == "2D"):
                data_HR = AvgPool2D(data_HR.unsqueeze(0), self.subsample_dist)[0]
                
        
        if(self.opt['random_flipping']):
            if(torch.rand(1).item() > 0.5):
                data_HR = torch.flip(data_HR,[1])
            if(torch.rand(1).item() > 0.5):
                data_HR = torch.flip(data_HR,[2])
            if(self.opt['mode'] == "3D"):
                if(torch.rand(1).item() > 0.5):
                    data_HR = torch.flip(data_HR,[3])
        return data_HR
    
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