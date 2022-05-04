from __future__ import absolute_import, division, print_function
import argparse
import time
import os

import matplotlib
from options import *
from netCDF4 import Dataset
import numpy as np
import imageio
from scipy.interpolate import RegularGridInterpolator  
import matplotlib.pyplot as plt
from numba import njit

@njit
def RK4(vf, positions, t0, h=0.5, direction="forward"):
    
    s = (int(positions.shape[0]), 1)
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

class interpolator_3D():
    def __init__(self, u_interpolator, v_interpolator):
        self.u_interpolator = u_interpolator
        self.v_interpolator = v_interpolator

    def __call__(self, points):
        u_values = self.u_interpolator(points)
        v_values = self.v_interpolator(points)
        interp_values = np.stack([v_values, u_values], axis=1)
        return interp_values

@njit
def particle_tracer(vf, positions, tstart = 0, tmax=100, h=0.5):
    yg = np.linspace(0, vf.shape[2]-1, vf.shape[2])
    xg = np.linspace(0, vf.shape[3]-1, vf.shape[3])
    tg = np.linspace(0, vf.shape[0]-1, vf.shape[0])
      
    u_interpolator = RegularGridInterpolator((tg,yg,xg), vf[:,1], 
                                             bounds_error=False, 
                                             fill_value=0)
    v_interpolator = RegularGridInterpolator((tg,yg,xg), vf[:,0], 
                                             bounds_error=False, 
                                             fill_value=0)
    interp = interpolator_3D(u_interpolator, v_interpolator)

    traces = []
    traces.append(positions.copy())
    
    for t in range(tstart, tstart+int(tmax / h)):
        t0 = t*h
        positions = RK4(
            np.concatenate([np.ones([positions.shape[0], 1])*t0,positions], axis=1), 
            interp, t0, h, "forward")[:,1:]
        traces.append(positions)
        
    return traces

@njit
def meshgrid(y, x):
    yy = np.empty(shape=(y.size, x.size), dtype=y.dtype)
    xx = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    for i in range(y.size):
        for j in range(x.size):
            yy[i,j] = i
            xx[i,j] = j
    return yy, xx

@njit             
def vf_to_flow_map(vf, t0, T, h=0.5, direction="forward"):
        
    yg = np.linspace(0, vf.shape[1]-1, vf.shape[1])
    xg = np.linspace(0, vf.shape[2]-1, vf.shape[2])
    g = np.stack(meshgrid(yg,xg))
    '''
    u_interpolator = RegularGridInterpolator((tg,yg,xg), vf[:,1], 
                                             bounds_error=False, 
                                             fill_value=0)
    v_interpolator = RegularGridInterpolator((tg,yg,xg), vf[:,0], 
                                             bounds_error=False, 
                                             fill_value=0)
    interp = interpolator_3D(u_interpolator, v_interpolator)
    '''

    t = t0    
    tmax = min(t0+T, vf.shape[0])

    positions = np.ascontiguousarray(g.copy().transpose((1,2,0))).reshape(-1, 2)
    while(t < tmax):
        #print(f"Tracing flow map at time {t}/{tmax}")
        s = (int(positions.shape[0]), 1)
        current_time = np.ones(s)*t
        interp_spots = np.concatenate((current_time,positions), axis=1)
        positions = RK4(
            vf, interp_spots,
            t0, h, "forward")[:,1:]
        t += h
    flow_map = np.ascontiguousarray(positions.transpose((1,0))).reshape(g.shape)
    return flow_map

#@njit
def FTLE_from_flow_map(fm, T):
    ftle = np.zeros((fm.shape[0], fm.shape[2], fm.shape[3]))
    
    for t in range(0,fm.shape[0]):
        print(f"Calculating FTLE for frame {t+1}/{fm.shape[0]}")
        dYdy = np.gradient(fm[t,0,:,:],axis=0)
        dYdx = np.gradient(fm[t,0,:,:],axis=1)
        dXdy = np.gradient(fm[t,1,:,:],axis=0)
        dXdx = np.gradient(fm[t,1,:,:],axis=1)

        full_gradient = np.array([[dYdy, dYdx],[dXdy, dXdx]])
        full_gradient = full_gradient.reshape(
            [full_gradient.shape[0], 
             full_gradient.shape[1], 
             -1])
        full_gradient = np.transpose(full_gradient, (2, 0, 1))
        sq_result = full_gradient.transpose((0,2,1)) @ full_gradient
        eigenvals = np.linalg.eigvals(sq_result)
        eigenvals = eigenvals.max(axis=1)
        eigenvals = eigenvals.reshape([fm.shape[2], fm.shape[3]])
        eigenvals = eigenvals ** 0.5
        vals = np.log(eigenvals) / T
        ftle[t] = vals
    
    return ftle
    
def vf_to_gif(vf, save_name):
    mag = np.linalg.norm(vf, axis=-1)
    print(mag.shape)
    
    mag -= mag.min()
    mag /= mag.max()
    mag *= 255
    mag = mag.astype(np.uint8)
    
    imageio.mimsave(save_name + ".gif", mag, fps=60)

def ftle_to_gif(ftle, save_name):
    ftle -= ftle.min()
    ftle /= (ftle.max()+1e-8)
    ftle *= 255
    ftle = ftle.astype(np.uint8)
    imageio.mimsave(save_name + ".gif", ftle, fps=60)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Calculate and save FTLE for data')
    parser.add_argument('--data_name',default=None,type=str,help='Input file')
    parser.add_argument('--save_name',default=None,type=str,help='Save folder name')
    
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    load_folder = os.path.join(project_folder_path, "Data", "FTLEFiles")
    save_folder = os.path.join(project_folder_path, "SuperResolutionData")

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

    vf_to_gif(vf[0:100], args['save_name'])
    
    # Make the vf in computation space
    vf[...,0] *= (vf.shape[1] / (0.5 + 0.5))
    vf[...,1] *= (vf.shape[2] / (7.5 + 0.5))
    vf *= (15 / vf.shape[0])
    
    '''
    # Check particle tracer
    traces = particle_tracer(vf, 
                             np.array(
                                 [[30, 5], 
                                  [50, 5],
                                  [40, 5],
                                  [39, 5],
                                  [37, 5]]),
                             tmax=1500,
                             tstart=600,
                             h=0.5)
    traces = np.stack(traces)
    print(traces.shape)
    fig,ax = plt.subplots()
    plt.scatter(traces[:,0,1], traces[:,0,0], color="green", s=2)
    plt.scatter(traces[:,1,1], traces[:,1,0], color="blue", s=2)
    plt.scatter(traces[:,2,1], traces[:,2,0], color="orange", s=2)
    plt.scatter(traces[:,3,1], traces[:,3,0], color="red", s=2)
    plt.scatter(traces[:,4,1], traces[:,4,0], color="gray", s=2)
    circle = plt.Circle((40, 40), radius=5, linewidth=0)
    c = matplotlib.collections.PatchCollection([circle], color='black')
    ax.add_collection(c)
    plt.title("Particle trace")
    plt.xlim([0, vf.shape[3]])
    plt.xlabel("x")
    plt.ylim([0, vf.shape[2]])
    plt.ylabel("y")
    plt.show()    
    '''
    
    T = 50
    h = 0.5
    skip = 25
    flow_maps = []
    for t0 in range(0, vf.shape[0]-T, skip):
        print(f"Calculting flow map {t0}/{vf.shape[0]}")
        t_start = time.time()
        fm = vf_to_flow_map(vf, t0, T, h)
        t_end = time.time()
        t_passed = t_end - t_start
        print(f"Calculation took {t_passed : 0.02f} seconds")
        flow_maps.append(fm)
    
    flow_maps = np.stack(flow_maps)
    
    ftle = FTLE_from_flow_map(flow_maps, T)

    ftle_to_gif(ftle, args['save_name']+"_ftle")
    