#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:18:10 2020

@author: hoss
"""
import tensorflow as tf
from deepcgh_new import DeepCGH_Datasets, DeepCGH
from time import time
import numpy as np
from glob import glob
import scipy.io as scio

# Define params
retrain = True
frame_path = 'DeepCGH_Frames/*.mat'
coordinates = False

data = {
        'path' : 'DeepCGH_Datasets/Disks',
        'shape' : (1152, 1920, 1),
        # 'shape' : (slm.height, slm.width, 3),
        'object_type' : 'Disk',
        'object_size' : 10,
        'object_count' : [27, 48],
        'intensity' : [0.2, 1],
        'normalize' : True,
        'centralized' : False,
        'N' : 10000,
        'train_ratio' : 9000/10000,
        'file_format' : 'tfrecords',
        'compression' : 'GZIP',
        'name' : 'target',
        }


model = {
        'path' : 'DeepCGH_Models/Disks',
        'int_factor':16,
        'n_kernels':[ 128, 256, 512],
        'plane_distance':0.005,
        'wavelength':1e-6,
        'pixel_size':0.000015,
        'input_name':'target',
        'output_name':'phi_slm',
        'lr' : 1e-3,
        'batch_size' : 4,
        'epochs' : 1,
        'token' : '',
        'max_steps' : 100
        }


# Get data
dset = DeepCGH_Datasets(data)

path = dset.getDataset()

# Estimator
dcgh = DeepCGH(data, model)

if retrain:
    dcgh.train(path, model['lr'], model['batch_size'], model['epochs'], model['token'], model['max_steps'])
    
#
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


