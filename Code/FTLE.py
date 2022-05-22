from __future__ import absolute_import, division, print_function
import argparse
import time
import os
import h5py
import matplotlib
from options import *
from netCDF4 import Dataset
import numpy as np
import imageio
import matplotlib.pyplot as plt
from numba import njit
from utility_functions import create_folder

@njit
def RK4(vf, positions, t0, h=0.5, direction="forward"):
    
    s = (positions.shape[0], 1)
    ones = np.ones(s)*t0

    k1 = interpolate(vf, positions)

    k2_spot = positions[:,1:] + (0.5 * k1 * h)
    k2_spot = np.concatenate(
        (ones+(0.5*h), 
        k2_spot), axis=1)
    k2 = interpolate(vf, k2_spot)

    k3_spot = positions[:,1:] + (0.5 * k2 * h)
    k3_spot = np.concatenate(
        (ones+(0.5*h), 
        k3_spot), axis=1)
    k3 = interpolate(vf, k3_spot)

    k4_spot = positions[:,1:] + (k3 * h)
    k4_spot = np.concatenate(
        (ones+h, 
        k4_spot), axis=1)
    k4 = interpolate(vf, k4_spot)
    
    positions[:,1:] += (1/6) * (k1+  2*k2 + 2*k3 + k4) * h
    positions[:,0] += h


    return positions

@njit
def Euler(vf, positions, t0, h=0.5, direction="forward"):
    
    vf_interp = interpolate(vf, positions)
    
    positions[:,1:] += vf_interp*h
    positions[:,0] += h

    return positions

@njit
def interpolate(data, positions):

    indices, weights = index_and_weights_for(positions)
    weights = np.repeat(weights, 2).reshape(positions.shape[0], 2*weights.shape[1])

    s = (positions.shape[0], data.shape[-1])
    values = np.zeros(s)
    
    max_values = [data.shape[0]-1], [data.shape[1]-1], [data.shape[2]-1]
    max_values = np.array(max_values, dtype=np.int32)

    max_values = np.repeat(max_values, int(positions.shape[0]))
    max_values = np.stack((max_values[0:positions.shape[0]], 
                        max_values[positions.shape[0]:2*positions.shape[0]], 
                        max_values[2*positions.shape[0]:]), axis=1)
    offset = np.zeros_like(max_values)

    values += weights[:,0:2] * solution_at(data, indices)
    
    offset[:,0] = 1
    values += weights[:,2:4] * solution_at(data, np.minimum(
        indices + offset,
        max_values))

    offset[:,0] = 0
    offset[:,1] = 1

    values += weights[:,4:6] * solution_at(data, np.minimum(
        indices + offset,
        max_values))
    offset[:,0] = 1

    values += weights[:,6:8] * solution_at(data, np.minimum(
        indices + offset,
        max_values))
    offset[:,0] = 0
    offset[:,1] = 0
    offset[:,2] = 1
    values += weights[:,8:10] * solution_at(data, np.minimum(
        indices + offset,
        max_values))
    offset[:,0] = 1
    values += weights[:,10:12] * solution_at(data, np.minimum(
        indices + offset,
        max_values))
    offset[:,0] = 0
    offset[:,1] = 1
    values += weights[:,12:14] * solution_at(data, np.minimum(
        indices + offset,
        max_values))
    offset[:,0] = 1
    values += weights[:,14:16] * solution_at(data, np.minimum(
        indices + offset,
        max_values))

    return values

@njit
def index_and_weights_for(positions):
    
    indices = positions 
    indices_floor = np.floor(indices.copy()).astype(np.int32)
    diffs = indices - indices_floor

    shape = (positions.shape[0], 8)
    weights = np.empty(shape)
    weights[:,0] = (1-diffs[:,0])*(1-diffs[:,1])*(1-diffs[:,2])
    weights[:,1] = diffs[:,0]*(1-diffs[:,1])*(1-diffs[:,2])
    weights[:,2] = (1-diffs[:,0])*diffs[:,1]*(1-diffs[:,2])
    weights[:,3] = diffs[:,0]*diffs[:,1]*(1-diffs[:,2])
    weights[:,4] = (1-diffs[:,0])*(1-diffs[:,1])*diffs[:,2]
    weights[:,5] = diffs[:,0]*(1-diffs[:,1])*diffs[:,2]
    weights[:,6] = (1-diffs[:,0])*diffs[:,1]*diffs[:,2]
    weights[:,7] = diffs[:,0]*diffs[:,1]*diffs[:,2]


    return indices_floor, weights

def solution_at_vectorized(data, positions):
    solutions = data[positions[:,0],
                    positions[:,1],
                    positions[:,2],
                    :]
    return solutions

@njit
def solution_at(data, positions):

    solutions = np.empty((positions.shape[0], data.shape[-1]))
    for i in range(positions.shape[0]):
        s = data[int(positions[i,0]), 
        int(positions[i,1]), 
        int(positions[i,2])]

        solutions[i] = s

    return solutions

@njit
def particle_tracer(vf, positions, tstart = 0.0, tfinish = 100.0, h = 0.5):
    traces = []
    traces.append(positions.copy())
    
    ts = np.arange(tstart, tfinish-h, h)
    for i in range(len(ts)):
        t = ts[i]
        s = (int(positions.shape[0]), 1)
        current_time = np.ones(s)*t
        interp_spots = np.concatenate((current_time,positions), axis=1)
        positions = RK4(vf,interp_spots, 
            t, h, "forward")[:,1:]
        traces.append(positions.copy())
        
    return traces

@njit
def meshgrid(y, x):
    yy = np.empty(shape=(y.size, x.size), dtype=y.dtype)
    xx = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    for i in range(y.size):
        for j in range(x.size):
            yy[i,j] = y[i]
            xx[i,j] = x[j]
    return yy, xx

@njit             
def vf_to_flow_map(vf, t0, T, h=0.5, direction="forward"):
        
    yg = np.linspace(0, vf.shape[1]-1, vf.shape[1])
    xg = np.linspace(0, vf.shape[2]-1, vf.shape[2])
    g = np.stack(meshgrid(yg,xg))
    
    t = t0    
    tmax = max(min(t0+T, vf.shape[0]), 0)
    
    positions = g.copy()
    positions = np.ascontiguousarray(positions.transpose((1,2,0)))
    positions = np.ascontiguousarray(positions.reshape(-1, 2))
    ts = np.arange(t0, tmax, h)
    for i in range(len(ts)):
        t = ts[i] 
        s = (int(positions.shape[0]), 1)
        current_time = np.ones(s)*t
        interp_spots = np.concatenate((current_time,positions), axis=1)
        positions = RK4(
            vf, interp_spots,
            t, h, "forward")[:,1:]
    
    
    flow_map = np.ascontiguousarray(positions)
    flow_map = np.ascontiguousarray(
        flow_map.reshape((vf.shape[1], vf.shape[2], vf.shape[3])))
    flow_map = np.ascontiguousarray(flow_map.transpose(2, 0, 1))
    return flow_map

#@njit
def FTLE_from_flow_map(fm, T):
    print("Calculating FTLE fields")
    ftle = np.zeros((fm.shape[0], fm.shape[2], fm.shape[3]))
    for t in range(0,fm.shape[0]):
        #print(f"Calculating FTLE for frame {t+1}/{fm.shape[0]}")
        dYdy = np.gradient(fm[t,0,:,:],axis=0, edge_order=2)
        dYdx = np.gradient(fm[t,0,:,:],axis=1, edge_order=2)
        dXdy = np.gradient(fm[t,1,:,:],axis=0, edge_order=2)
        dXdx = np.gradient(fm[t,1,:,:],axis=1, edge_order=2)

        full_gradient = np.array([[dYdy, dYdx],[dXdy, dXdx]])
        full_gradient = full_gradient.reshape(
            [full_gradient.shape[0], 
             full_gradient.shape[1], 
             -1])
        full_gradient = np.ascontiguousarray(np.transpose(full_gradient, (2, 0, 1)))
        sq_result = full_gradient.copy().transpose((0,2,1)) @ full_gradient
        eigenvals = np.linalg.eigvals(sq_result)
        eigenvals = eigenvals.max(axis=1)
        eigenvals = eigenvals.reshape([fm.shape[2], fm.shape[3]])
        eigenvals = eigenvals ** 0.5
        vals = np.log(eigenvals) / T
        ftle[t] = vals
    
    return ftle
    
def vf_to_gif(vf, save_name):
    mag = np.linalg.norm(vf, axis=-1)
    
    mag -= mag.min()
    mag /= mag.max()
    mag *= 255
    mag = mag.astype(np.uint8)
    mag = np.flip(mag, axis=1)
    
    imageio.mimsave(save_name + ".gif", mag, fps=60)

def ftle_to_gif(ftle, save_name):
    ftle -= ftle.min()
    ftle /= (ftle.max()+1e-8)
    ftle *= 255
    ftle = ftle.astype(np.uint8)
    ftle = np.flip(ftle, axis=1)
    imageio.mimsave(save_name + ".gif", ftle, fps=60)

def save_FTLE_data(ftle, save_folder, prefix):
    ftle -= ftle.min()
    ftle /= (ftle.max()+1e-8)
    ftle = np.flip(ftle, axis=1)
    for i in range(ftle.shape[0]):
        f = h5py.File(os.path.join(save_folder, prefix+"_"+str(i)+".h5"), mode='w')
        f.create_dataset("data", data=np.expand_dims(ftle[i], 0))
        f.close()
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Calculate and save FTLE for data')
    parser.add_argument('--data_name',default=None,type=str,help='Input file')
    parser.add_argument('--save_name',default=None,type=str,help='Save folder name')
    parser.add_argument('--T',default=10,type=float,help='Advection time')
    parser.add_argument('--h',default=0.5,type=float,help='Advection step')
    parser.add_argument('--skip',default=1,type=int,help='timestep skip')
    
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    load_folder = os.path.join(project_folder_path, "Data", "FTLEFiles")
    save_folder = os.path.join(project_folder_path, "Data", "SuperResolutionData")

    nc_path = os.path.join(load_folder, args['data_name'])
    d = Dataset(nc_path, "r")
    print(d)

    u = np.array(d['u'])
    v = np.array(d['v'])
    
    d.close()
    vf = np.stack([v,u], axis=0)
    print(vf.shape)
    vf = np.transpose(vf, (1, 2, 3, 0))    
    print(vf.shape)

    vf_to_gif(vf[0:1000], args['save_name'])
    
    # Make the vf in computation space
    #vf[...,0] *= (vf.shape[1] / (3.0))
    #vf[...,1] *= (vf.shape[2] / (1.0))
    #vf *= (20.0 / vf.shape[0])
    
    '''
    # Check particle tracer
    yg = np.linspace(0, vf.shape[1]-1, int(vf.shape[1]))
    xg = np.linspace(0, vf.shape[2]-1, int(vf.shape[2]/128))
    g = np.stack(meshgrid(yg,xg))
    g = g[:,30:50,0:1]
    t = 0    
    tmax = max(min(0+15, vf.shape[0]), 0)
    
    positions = g.copy()
    positions = np.ascontiguousarray(positions.transpose((1,2,0)))
    positions = np.ascontiguousarray(positions.reshape(-1, 2))
    traces = particle_tracer(vf, 
                             positions,
                             tstart=1000.0,
                             tfinish=1500.0,
                             h=0.05)
                             
    traces = np.stack(traces)
    print(traces.shape)
    fig,ax = plt.subplots()
    for i in range(traces.shape[1]):        
        plt.scatter(traces[:,i,1], traces[:,i,0])
    circle = plt.Circle((40, 40), radius=5, linewidth=0)
    c = matplotlib.collections.PatchCollection([circle], color='black')
    ax.add_collection(c)
    plt.title("Particle trace")
    plt.xlim([0, vf.shape[2]])
    plt.xlabel("x")
    plt.ylim([0, vf.shape[1]])
    plt.ylabel("y")
    plt.show()  
    '''
    
    #vf = vf[:,::4, ::4, :]
    
    #T = args['T']
    h = args['h']
    #skip = args['skip']
    total_flow_map_calculation = 0
    total_ftle_calculation = 0
    Ts = [1, 5, 10, 25, 50, 100, 250, 500]
    for T_spot in range(len(Ts)):
    #for T in range(50, 51, 5):
        T = Ts[T_spot]
        flow_maps = []
        for t0 in np.arange(max(0.0, 0.0-T), min(vf.shape[0], vf.shape[0]-T)-1, 1):
        #for t0 in np.arange(5, 1000, 5):
            print(f"Calculting flow map {t0}/{vf.shape[0]}")
            t_start = time.time()
            fm = vf_to_flow_map(vf, t0, T, h)
            t_end = time.time()
            t_passed = t_end - t_start
            total_flow_map_calculation += t_passed
            print(f"Calculation took {t_passed : 0.02f} seconds")
            flow_maps.append(fm)
        flow_maps = np.stack(flow_maps)

        t_start = time.time()
        ftle = FTLE_from_flow_map(flow_maps, T)    
        t_end = time.time()
        t_passed = t_end - t_start
        total_ftle_calculation = t_passed
        #ftle_to_gif(ftle, args['save_name']+"_ftle")
        
        create_folder(save_folder, args['save_name'])
        save_FTLE_data(ftle, os.path.join(save_folder, 
                        args['save_name']), 
                        str(T)+"_"+str(h))
    
    print(f"Flow map stats - total:{total_flow_map_calculation : 0.04f} average:{total_flow_map_calculation/ftle.shape[0] : 0.04f}")
    print(f"FTLE stats - total:{total_ftle_calculation : 0.04f} average:{total_ftle_calculation/ftle.shape[0] : 0.04f}")
    print(f"Total stats - total:{total_flow_map_calculation+total_ftle_calculation : 0.04f} average:{(total_flow_map_calculation+total_ftle_calculation)/ftle.shape[0] : 0.04f}")