#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:18:10 2020

@author: hoss
"""
import tensorflow as tf
from deepcgh import DeepCGH_Datasets, DeepCGH
import numpy as np
from glob import glob
import scipy.io as scio
from utils import GS3D, display_results, get_propagate
# Define params
retrain = True
frame_path = 'DeepCGH_Frames/*.mat'
coordinates = False

data = {
        'path' : 'DeepCGH_Datasets/Disks',
        'shape' : (512, 512, 3),
        # 'shape' : (slm.height, slm.width, 3),
        'object_type' : 'Disk',
        'object_size' : 10,
        'object_count' : [27, 48],
        'intensity' : [0.2, 1],
        'normalize' : True,
        'centralized' : False,
        'N' : 20000,
        'train_ratio' : 19000/20000,
        'compression' : 'GZIP',
        'name' : 'target',
        }


model = {
        'path' : 'DeepCGH_Models/Disks',
        'int_factor':32,
        'n_kernels':[ 64, 128, 256],
        'plane_distance':0.005,
        'wavelength':1e-6,
        'pixel_size':0.000015,
        'input_name':'target',
        'output_name':'phi_slm',
        'lr' : 1e-4,
        'batch_size' : 4,
        'epochs' : 10,
        'token' : '64',
        'shuffle' : 8,
        'max_steps' : 4000
        }


# Get data
dset = DeepCGH_Datasets(data)

dset.getDataset()

# Estimator
dcgh = DeepCGH(data, model)

if retrain:
    dcgh.train(dset)
    
#%%
while(True):
    files = glob(frame_path)
    if len(files) > 0:
        data = scio.loadmat(files[0])['data']
        if data.sum() == 0:
            break
        if coordinates:
            data = dset.coord2image(data)
    phase = np.squeeze(dcgh.get_hologram(data))

    # you can add your code here


#%% This is a sample test. You can generate a random image and get the results
propagate = get_propagate(data, model)
image = dset.get_randSample()[np.newaxis,...]
# making inference is as simple as calling the get_hologram method
phase = dcgh.get_hologram(image)
propagate = get_propagate(data, model)
reconstruction = propagate(phase)
display_results(image, phase, reconstruction, 1)