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
        'shape' : (1152, 1920, 5),
        # 'shape' : (slm.height, slm.width, 3),
        'object_type' : 'Disk',
        'object_size' : 10,
        'object_count' : [27, 48],
        'intensity' : [0.1, 1],
        'normalize' : True,
        'centralized' : False,
        'N' : 20,
        'train_ratio' : 10/20,
        'compression' : 'GZIP',
        'name' : 'target',
        }


model = {
        'path' : 'DeepCGH_Models/Disks',
        'int_factor':16,
        'n_kernels':[64, 128, 256],
        'plane_distance':19.3/1000,
        'focal_point':0.2,
        # 'wavelength':1.e-6,
        # 'pixel_size':0.000015,
        'wavelength':1.04e-6,
        'pixel_size': 9.2e-6,
        'input_name':'target',
        'output_name':'phi_slm',
        'lr' : 1e-4,
        'batch_size' : 16,
        'epochs' : 100,
        'token' : 'final',
        'shuffle' : 16,
        'max_steps' : 4000,
        # 'HMatrix' : hstack
        }


# Get data
dset = DeepCGH_Datasets(data)

dset.getDataset()

# Estimator
dcgh = DeepCGH(data, model)

if retrain:
    dcgh.train(dset)
    
#%% Example inifinity loop
# while(True):
#     files = glob(frame_path)
#     if len(files) > 0:
#         data = scio.loadmat(files[0])['data']
#         if data.sum() == 0:
#             break
#         if coordinates:
#             data = dset.coord2image(data)
#     phase = np.squeeze(dcgh.get_hologram(data))


#%% This is a sample test. You can generate a random image and get the results
model['HMatrix'] = dcgh.Hs # For plotting we use the exact same H matrices that DeepCGH used

# Get a function that propagates SLM phase to different planes according to your setup's characteristics
propagate = get_propagate(data, model)

# Generate a random sample
image = dset.get_randSample()[np.newaxis,...]
# Get the phase for your target using a trained and loaded DeepCGH
phase = dcgh.get_hologram(image)

# Simulate what the solution would look like
reconstruction = propagate(phase)

# Show the results
display_results(image, phase, reconstruction, 1)