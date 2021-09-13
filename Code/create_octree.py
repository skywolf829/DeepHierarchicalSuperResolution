import torch
import torch.nn.functional as F
import torch.jit
from utility_functions import AvgPool2D, AvgPool3D
import time
from typing import Dict, List
import h5py
import argparse
import os

class OctreeNode:
    def __init__(self, data : torch.Tensor, 
    LOD : int, depth : int, index : int):
        self.data : torch.Tensor = data 
        self.LOD : int = LOD
        self.depth : int = depth
        self.index : int = index

    def __str__(self) -> str:
        return "{ data_shape: " + str(self.data.shape) + ", " + \
        "LOD: " + str(self.LOD) + ", " + \
        "depth: " + str(self.depth) + ", " + \
        "index: " + str(self.index) + "}" 
        
    def min_width(self) -> int:
        m = self.data.shape[2]
        for i in range(3, len(self.data.shape)):
            m = min(m, self.data.shape[i])
        return m

    def size(self) -> float:
        return (self.data.element_size() * self.data.numel()) / 1024.0

class OctreeNodeList:
    def __init__(self):
        self.node_list : List[OctreeNode] = []

    def append(self, n : OctreeNode):
        self.node_list.append(n)

    def insert(self, i : int, n: OctreeNode):
        self.node_list.insert(i, n)

    def pop(self, i : int) -> OctreeNode:
        return self.node_list.pop(i)

    def remove(self, item : OctreeNode) -> bool:
        found : bool = False
        i : int = 0
        while(i < len(self.node_list) and not found):
            if(self.node_list[i] is item):
                self.node_list.pop(i)
                found = True
            i += 1
        return found
    
    def __len__(self) -> int:
        return len(self.node_list)

    def __getitem__(self, key : int) -> OctreeNode:
        return self.node_list[key]

    def __str__(self):
        s : str = "["
        for i in range(len(self.node_list)):
            s += str(self.node_list[i])
            if(i < len(self.node_list)-1):
                s += ", "
        s += "]"
        return s

    def total_size(self):
        nbytes = 0.0
        for i in range(len(self.node_list)):
            nbytes += self.node_list[i].size()
        return nbytes 
    
    def mean(self):
        m = torch.zeros([self.node_list[0].data.shape[1]])
        dims = [0, 2, 3]
        if(len(self.node_list[0].data.shape) == 5):
            dims.append(4)
        for i in range(len(self.node_list)):
            m += self.node_list[i].data.mean(dims).cpu()
        return m / len(self.node_list)

    def max(self):
        m = self.node_list[0].data.max().cpu().item()
        for i in range(len(self.node_list)):
            m = max(m, self.node_list[i].data.max().cpu().item())
        return m

    def min(self):
        m = self.node_list[0].data.min().cpu().item()
        for i in range(len(self.node_list)):
            m = min(m, self.node_list[i].data.min().cpu().item())
        return m
    
    def num_voxels(self):
        total = 0
        for i in range(len(self.node_list)):
            this_total = 1
            for j in range(2, len(self.node_list[i].data.shape)):
                this_total *= self.node_list[i].data.shape[j]
            total += this_total
        return total

    def max_LOD(self):
        max_lod = self.node_list[0].LOD
        for i in range(len(self.node_list)):
            max_lod = max(max_lod, self.node_list[i].LOD)
        return max_lod

def split_node(n):
    nodes = []
    k = 0
    for x_quad_start in range(0, n.data.shape[2], int(n.data.shape[2]/2)):
        for y_quad_start in range(0, n.data.shape[3], int(n.data.shape[3]/2)):
            if(len(n.data.shape) == 5):
                for z_quad_start in range(0, n.data.shape[4], int(n.data.shape[4]/2)):
                    n_quad = OctreeNode(
                        n.data[:,:,
                            x_quad_start:x_quad_start+int(n.data.shape[2]/2),
                            y_quad_start:y_quad_start+int(n.data.shape[3]/2),
                            z_quad_start:z_quad_start+int(n.data.shape[4]/2)].clone(),
                        n.LOD,
                        n.depth+1,
                        n.index*8 + k
                    )
                    nodes.append(n_quad)
                    k += 1     
            else:
                n_quad = OctreeNode(
                    n.data[:,:,
                        x_quad_start:x_quad_start+int(n.data.shape[2]/2),
                        y_quad_start:y_quad_start+int(n.data.shape[3]/2)].clone(),
                    n.LOD,
                    n.depth+1,
                    n.index*4 + k
                )
                nodes.append(n_quad)
                k += 1      
    return nodes 

def voxels_at_each_LOD(nodes):
    voxel_breakdown = {}
    node_breakdown = {}
    for i in range(len(nodes)):
        n = nodes[i]
        if(n.LOD not in voxel_breakdown):
            voxel_breakdown[n.LOD] = 0
            node_breakdown[n.LOD] = 0
        n_voxels = 1
        for j in range(2, len(n.data.shape)):
            n_voxels *= n.data.shape[j]
        voxel_breakdown[n.LOD] += n_voxels
        node_breakdown[n.LOD] += 1
    return voxel_breakdown, node_breakdown

def downscale(data):
    if(len(data.shape) == 4):
        return AvgPool2D(data, 2)
    else:
        return AvgPool3D(data, 2)

def volume_to_octree(volume, epsilon, min_chunk, max_downscaling_level):
    root = OctreeNode(volume.clone(), 0, 0, 0)
    octree = OctreeNodeList()
    octree.append(root)

    queue = []
    queue.append(root)

    while(len(queue) > 0):
        node = queue.pop(0)
        if(node.LOD < max_downscaling_level and node.min_width() >= min_chunk):
            node_downscaled = downscale(node.data.clone())
            node_downscaled_test = F.interpolate(node_downscaled, mode='nearest', scale_factor=2)
            node_downscaled_test -= node.data
            if(torch.all(torch.abs(node_downscaled_test) < epsilon)):
                node.LOD += 1
                node.data = node_downscaled
                queue.append(node)
            else:
                node_split = split_node(node)
                for n in node_split:
                    octree.append(n)
                    queue.append(n)
                octree.remove(node)

    return octree      

def join_redundant_octree_nodes(nodes: OctreeNodeList) -> OctreeNodeList:

    device = nodes[0].data.device
    current_depth = nodes[0].depth

    # dict[depth -> LOD -> group parent index -> list]
    groups : Dict[int, Dict[int, Dict[int, Dict[int, OctreeNode]]]] = {}

    magic_num : int = 4 if len(nodes[0].data.shape) == 4 else 8
    for i in range(len(nodes)):
        d : int = nodes[i].depth
        l : int = nodes[i].LOD
        group_parent_index : int = int(nodes[i].index / magic_num)
        n_index : int = nodes[i].index % magic_num
        current_depth = max(current_depth, d)
        if(d not in groups.keys()):
            groups[d] = {}
        if(l not in groups[d].keys()):
            groups[d][l] = {}
        if(group_parent_index not in groups[d][l].keys()):
            groups[d][l][group_parent_index] = {}
        groups[d][l][group_parent_index][n_index] = nodes[i]

            
    while(current_depth  > 0):
        if(current_depth in groups.keys()):
            for lod in groups[current_depth].keys():
                for parent in groups[current_depth][lod].keys():
                    group = groups[current_depth][lod][parent]
                    if(len(group) == magic_num):
                        if(len(group[0].data.shape) == 4):
                            new_data = torch.zeros([
                                group[0].data.shape[0],
                                group[0].data.shape[1],
                                group[0].data.shape[2]*2, 
                                group[0].data.shape[3]*2], device=device, 
                            dtype=group[0].data.dtype)
                            new_data[:,:,:group[0].data.shape[2],
                                    :group[0].data.shape[3]] = \
                                group[0].data

                            new_data[:,:,:group[0].data.shape[2],
                                    group[0].data.shape[3]:] = \
                                group[1].data

                            new_data[:,:,group[0].data.shape[2]:,
                                    :group[0].data.shape[3]] = \
                                group[2].data

                            new_data[:,:,group[0].data.shape[2]:,
                                    group[0].data.shape[3]:] = \
                                group[3].data
                            
                            new_node = OctreeNode(new_data, group[0].LOD, 
                            group[0].depth-1, parent)
                            nodes.append(new_node)
                            nodes.remove(group[0])
                            nodes.remove(group[1])
                            nodes.remove(group[2])
                            nodes.remove(group[3])
                            d = current_depth-1
                            l = lod
                            if(d not in groups.keys()):
                                groups[d] = {}
                            if(lod not in groups[d].keys()):
                                groups[d][l] = {}
                            if(int(parent/4) not in groups[d][l].keys()):
                                groups[d][l][int(parent/4)] = {}
                            groups[d][l][int(parent/4)][new_node.index % 4] = new_node
                        else:
                            new_data = torch.zeros([
                                group[0].data.shape[0],
                                group[0].data.shape[1],
                                group[0].data.shape[2]*2, 
                                group[0].data.shape[3]*2,
                                group[0].data.shape[4]*2], device=device, 
                            dtype=group[0].data.dtype)
                            new_data[:,:,
                                    :group[0].data.shape[2],
                                    :group[0].data.shape[3],
                                    :group[0].data.shape[4]] = \
                                group[0].data

                            new_data[:,:,
                                    :group[0].data.shape[2],
                                    :group[0].data.shape[3],
                                    group[0].data.shape[4]:] = \
                                group[1].data

                            new_data[:,:,
                                    :group[0].data.shape[2],
                                    group[0].data.shape[3]:,
                                    :group[0].data.shape[4]] = \
                                group[2].data

                            new_data[:,:,
                                    :group[0].data.shape[2],
                                    group[0].data.shape[3]:,
                                    group[0].data.shape[4]:] = \
                                group[3].data

                            new_data[:,:,
                                    group[4].data.shape[2]:,
                                    :group[0].data.shape[3],
                                    :group[0].data.shape[4]] = \
                                group[4].data

                            new_data[:,:,
                                    group[0].data.shape[2]:,
                                    :group[0].data.shape[3],
                                    group[0].data.shape[4]:] = \
                                group[5].data

                            new_data[:,:,
                                    group[0].data.shape[2]:,
                                    group[0].data.shape[3]:,
                                    :group[0].data.shape[4]] = \
                                group[6].data

                            new_data[:,:,
                                    group[0].data.shape[2]:,
                                    group[0].data.shape[3]:,
                                    group[0].data.shape[4]:] = \
                                group[7].data
                            
                            new_node = OctreeNode(new_data, group[0].LOD, 
                            group[0].depth-1, int(group[0].index / 8))
                            nodes.append(new_node)
                            nodes.remove(group[0])
                            nodes.remove(group[1])
                            nodes.remove(group[2])
                            nodes.remove(group[3])
                            nodes.remove(group[4])
                            nodes.remove(group[5])
                            nodes.remove(group[6])
                            nodes.remove(group[7])
                            d = current_depth-1
                            l = lod
                            if(d not in groups.keys()):
                                groups[d] = {}
                            if(lod not in groups[d].keys()):
                                groups[d][l] = {}
                            if(int(parent/8) not in groups[d][l].keys()):
                                groups[d][l][int(parent/8)] = {}
                            groups[d][l][int(parent/8)][new_node.index % 8] = new_node
                        
        current_depth -= 1
    return nodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')

    parser.add_argument('--file',default="Plume.h5",type=str,help='File to octree-ify. Should be of shape [c, h, w, [d]]')
    parser.add_argument('--save_name',default="Plume.octree",type=str,help='Name for trial in results.pkl')
    parser.add_argument('--epsilon',default=0.1,type=float,help='PSNR to start tests at')
    
    parser.add_argument('--max_downscaling_level',default=3,type=int,help="The maximum downscaling level to support in the created octree")
    parser.add_argument('--min_chunk',default=2,type=int,help="Minimum block size to reduce")
    parser.add_argument('--device',default="cuda:0",type=str)

    args = vars(parser.parse_args())
    
    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data", "FilesToOctreeify")
    save_folder = os.path.join(project_folder_path, "Data", "OctreeFiles")
    file_path = os.path.join(data_folder, args['file'])
            
    print("Loading " + file_path)   
    f = h5py.File(file_path, 'r')
    d = torch.tensor(f.get('data'))
    f.close()
    volume = d.unsqueeze(0).to(args['device'])

    print("Data range: %0.02f - %0.02f, mean: %0.02f" % (volume.min(), volume.max(), volume.mean()))
    print("Octreeifying the volume of size " + str(volume.shape))
    start_time = time.time()
    octree = volume_to_octree(volume, args['epsilon'], args['min_chunk'], args['max_downscaling_level'])
    end_time = time.time()
    print("Octreeification took %0.02f seconds" % (end_time-start_time))

    print("")
    print("########################### BEFORE JOINING REDUNDANT NODES #############################################")
    print("")

    print("The octree with epsilon=%0.02f, max_downscaling_level=%i, and min_chunk=%i has %i octree leaf nodes" % \
        (args['epsilon'], args['max_downscaling_level'], args['min_chunk'], len(octree)))
    total_voxels = 1
    for i in range(2, len(volume.shape)):
        total_voxels *= volume.shape[i]
    print("The octree has %i voxels, which is %0.02f percent of the original data" % \
        (octree.num_voxels(), 100* octree.num_voxels() / total_voxels))
    
    voxel_breakdown, node_breakdown = voxels_at_each_LOD(octree)

    lods = list(voxel_breakdown.keys())
    lods.sort()
    for i in range(len(lods)):
        print("Voxels at downscaling level %i: %i (%0.02f percent)" % (lods[i], voxel_breakdown[lods[i]], 100*voxel_breakdown[lods[i]] / octree.num_voxels()))
        print("Nodes at downscaling level %i: %i (%0.02f percent)" % (lods[i], node_breakdown[lods[i]], 100*node_breakdown[lods[i]] / len(octree)))


    print("")
    print("########################### AFTER JOINING REDUNDANT NODES #############################################")
    print("")

    print("Joining redundant nodes")
    start_time = time.time()
    octree = join_redundant_octree_nodes(octree)
    end_time = time.time()
    print("Joining redundant octree nodes took %0.02f seconds" % (end_time - start_time))

    print("The octree with epsilon=%0.02f, max_downscaling_level=%i, and min_chunk=%i has %i octree leaf nodes" % \
        (args['epsilon'], args['max_downscaling_level'], args['min_chunk'], len(octree)))
    total_voxels = 1
    for i in range(2, len(volume.shape)):
        total_voxels *= volume.shape[i]
    print("The octree has %i voxels, which is %0.02f percent of the original data" % \
        (octree.num_voxels(), 100* octree.num_voxels() / total_voxels))
    
    voxel_breakdown, node_breakdown = voxels_at_each_LOD(octree)

    lods = list(voxel_breakdown.keys())
    lods.sort()
    for i in range(len(lods)):
        print("Voxels at downscaling level %i: %i (%0.02f percent)" % (lods[i], voxel_breakdown[lods[i]], 100*voxel_breakdown[lods[i]] / octree.num_voxels()))
        print("Nodes at downscaling level %i: %i (%0.02f percent)" % (lods[i], node_breakdown[lods[i]], 100*node_breakdown[lods[i]] / len(octree)))

    torch.save(octree, os.path.join(save_folder, args['save_name']))
    print("The octree data was saved to " + os.path.join(save_folder, args['save_name']))