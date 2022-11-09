#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:18:10 2020

@author: hoss
"""
import tensorflow as tf
from deepcgh import DeepCGH_Datasets, DeepDCGH
import numpy as np
from glob import glob
import scipy.io as scio
from utils import GS3D, display_results, get_propagate
import matplotlib.pyplot as plt

# Define params
retrain = True
frame_path = 'DeepCGH_Frames/*.mat'
coordinates = False

data = {
        'path' : 'DeepCGH_Datasets/Disks',
        'shape' : (512, 512, 3),
        'object_type' : 'Disk',
        'object_size' : 10,
        'object_count' : [27, 48],
        'intensity' : [0.1, 1],
        'normalize' : True,
        'centralized' : False,
        'N' : 2000,
        'train_ratio' : 1900/2000,
        'compression' : 'GZIP',
        'name' : 'target',
        }


model = {
        'path' : 'DeepCGH_Models/Disks',
        'num_frames':5,
        'quantization':6,
        'int_factor':16,
        'n_kernels':[64, 128, 256],
        'plane_distance':0.05,
        'focal_point':0.2,
        'wavelength':1.04e-6,
        'pixel_size': 9.2e-6,
        'input_name':'target',
        'output_name':'phi_slm',
        'lr' : 1e-4,
        'batch_size' : 16,
        'epochs' : 100,
        'token' : 'DCGH',
        'shuffle' : 16,
        'max_steps' : 4000,
        # 'HMatrix' : hstack
        }


# Get data
dset = DeepCGH_Datasets(data)

dset.getDataset()

# Estimator
dcgh = DeepDCGH(data, model)

if retrain:
    dcgh.train(dset)

#%% This is a sample test. You can generate a random image and get the results
model['HMatrix'] = dcgh.Hs # For plotting we use the exact same H matrices that DeepCGH used

# Get a function that propagates SLM phase to different planes according to your setup's characteristics
propagate = get_propagate(data, model)

# Generate a random sample
image = dset.get_randSample()[np.newaxis,...]
# Get the phase for your target using a trained and loaded DeepCGH
phase = dcgh.get_hologram(image)

# Simulate what the solution would look like
reconstruction = propagate(phase).numpy()

#%% display simulation results
plt.figure(figsize=(30, 20))
Z = [-50, 0, 50]
for i in range(reconstruction.shape[-1]):
    plt.subplot(231+i)
    plt.imshow(reconstruction[0, :,:, i], cmap='gray')
    plt.axis('off')
    plt.title('Simulation @ {}mm'.format(Z[i]))
    plt.subplot(234+i)
    plt.imshow(image[0, :,:, i], cmap='gray')
    plt.axis('off')
    plt.title('Target @ {}mm'.format(Z[i]))
plt.savefig('example.jpg')
plt.show()

#%%
