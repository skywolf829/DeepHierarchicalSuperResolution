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

def RK4(positions, interpolator, t0, h=0.5, direction="forward"):
    k1 = interpolator(positions)
    k2_spot = positions[:,1:] + 0.5 * k1 * h
    k2_spot = np.concatenate(
        [np.ones([positions.shape[0], 1])*t0+0.5*h, 
        k2_spot], axis=1)
    k2 = interpolator(k2_spot)
    k3_spot = positions[:,1:] + 0.5 * k2 * h
    k3_spot = np.concatenate(
        [np.ones([positions.shape[0], 1])*t0+0.5*h, 
        k3_spot], axis=1)
    k3 = interpolator(k3_spot)
    k4_spot = positions[:,1:] + k3 * h
    k4_spot = np.concatenate(
        [np.ones([positions.shape[0], 1])*t0+h, 
        k4_spot], axis=1)
    k4 = interpolator(k4_spot)
    
    positions[:,1:] += (1/6) * (k1+  2*k2 + 2*k3 + k4) * h
    positions[:,0] += h

    return positions

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
    
    def interpolator(points):
        u_values = u_interpolator(points)
        v_values = v_interpolator(points)
        interp_values = np.stack([v_values, u_values], axis=1)
        return interp_values

    traces = []
    traces.append(positions.copy())
    
    for t in range(tstart, tstart+int(tmax / h)):
        t0 = t*h
        positions = RK4(
            np.concatenate([np.ones([positions.shape[0], 1])*t0,positions], axis=1), 
            interpolator, t0, h, "forward")[:,1:]
        traces.append(positions)
        
    return traces
               
def vf_to_flow_map(vf, t0, T, h=0.5, direction="forward"):
        
    yg = np.linspace(0, vf.shape[2]-1, vf.shape[2])
    xg = np.linspace(0, vf.shape[3]-1, vf.shape[3])
    tg = np.linspace(0, vf.shape[0]-1, vf.shape[0])
    g = np.stack(np.meshgrid(yg,xg,indexing="ij"))
      
    u_interpolator = RegularGridInterpolator((tg,yg,xg), vf[:,1], 
                                             bounds_error=False, 
                                             fill_value=0)
    v_interpolator = RegularGridInterpolator((tg,yg,xg), vf[:,0], 
                                             bounds_error=False, 
                                             fill_value=0)
    
    def interpolator(points):
        u_values = u_interpolator(points)
        v_values = v_interpolator(points)
        interp_values = np.stack([v_values, u_values], axis=1)
        return interp_values
    
    t = t0    
    tmax = min(t0+T, vf.shape[0])

    positions = g.copy().transpose((1,2,0)).reshape(-1, 2)
    while(t < tmax):
        #print(f"Tracing flow map at time {t}/{tmax}")
        positions = RK4(
            np.concatenate([np.ones([positions.shape[0], 1])*t,positions], axis=1), 
            interpolator, t0, h, "forward")[:,1:]
        t += h
    flow_map = positions.transpose((1,0)).reshape(g.shape)
    
    '''
    for t in range(t0, tmax):
        print(f"Computing flow map for timestep {t}/{vf.shape[0]}")
        points_start = flow_map[t].transpose(1,2,0).reshape(-1, 2)
        points_start = np.concatenate(
            [np.zeros([points_start.shape[0], 1]), 
            points_start], axis=1)
        points_start[:,0] = t
        
        for i in range(interp_per_timestep):
            points_start = RK4(points_start, interpolator, 
                               t+i, 0.5, "forward")
            
        flow_map[t+1] = points_start[:,1:].transpose(1,0).reshape(flow_map.shape[1:])
    '''
    
    return flow_map

def FTLE_from_flow_map(fm, T):
    ftle = np.zeros([fm.shape[0], fm.shape[2], fm.shape[3]])
    
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
    mag = np.linalg.norm(vf, axis=1)
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
    vf = np.transpose(vf, (1, 0, 2, 3))    
    print(vf.shape)
    vf_to_gif(vf[0:100], args['save_name'])
    
    # Make the vf in computation space
    vf[:,0] *= (vf.shape[2] / (0.5 + 0.5))
    vf[:,1] *= (vf.shape[3] / (7.5 + 0.5))
    vf *= (15 / 1501)
    
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
    flow_maps = []
    for t0 in range(0, vf.shape[0]-T, 25):
        print(f"Calculting flow map {t0}/{vf.shape[0]}")
        t_start = time.time()
        fm = vf_to_flow_map(vf, t0, T, h)
        t_elapsed = time.time() - t_start
        print(f"Calculation took {t_elapsed : 0.02f} seconds")
        flow_maps.append(fm)
    
    flow_maps = np.stack(flow_maps)
    
    ftle = FTLE_from_flow_map(flow_maps, T)
    ftle_to_gif(ftle, args['save_name']+"_ftle")
    