![header](./header.png)
Format: ![Alt Text](url)

# DeepCGH: 3D computer generated holography using deep learning
DeepCGH is an *unsupervised*, *non-iterative* algorithm for computer generated holography. DeepCGH relies on convolutional neural networks to perform *image plane* holography in real-time.
For more details regarding the structure and algorithm refer to the associated manuscript [1].

## Installation Guide and Dependencies:
Here we provide a Python and Tensorflow implementation of DeepCGH. The current version of this software does not require explciit installation. Dependencies include:
```
python 3.x
tensorflow-gpu >= 2.0.0
h5py
scipy
skimage
tqdm
```
If you have Python 3.x (preferrably > 3.7), you can easily install package requirements by executing the following command in **Ubuntu** terminal or Anaconda Prompt on **Windows**:
```
pip install tensorflow-gpu==2.3.0 h5py scipy skimage tqdm
```
This software was not tested on **Mac OS** but theoretically it should run smoothly independent of the OS.
After the installation of packages is complete, you can clone this repository to your machine using:
```git clone https://github.com/UNC-optics/DeepCGH.git```

## Usage
### For Users
After cloning the repository (see previous section), you can run the demo `demo_infinity_loop.py` for a simple inifinity loop example. In this demo:

First, the parameters of the simulated training dataset are determined:
```
data = {
        'path' : 'DeepCGH_Datasets/Disks',
        'shape' : (1152, 1920, 1),
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
```
Please note that you can also provide other datasets (such as natural images) as training data (instructions coming soon).
The parameters in this dictionary are:
1. `'shape'` : The spatial dimensions and number of depth planes of the holograms are determined by the `shape` parameter. The convention is (y_dimension, x_dimention, number_of_planes).
2. `'path'` : determines the directory at which the dataset file is stored
3. `'object_type'` : either `'Disk'`, `'Line'`, or `'Dot'`. Determines what kind of object should be randomly generated.
4. `'object_size'` : determines the sthe radius of Disks. If `object_type!='Disk'` then anything work. Do not leave empty.
5. `'object_count'` : determines the min and max number of objects per plane. Must be a list
6. `'intensity'` : determines the min and max of the intensity of each object. Intensities will be determined randomly.
7. `'normalize'` : a flag that determines whether the intensoty of depth planes are normalized with respect to each other. If True, the energy in each depth plane is equal.
8. `'centralized'` : focuses the location of objects in the center. This is a practical feature for real SLMs.
9. `'N'` : the number of samples
10. `'train_ratio'` : determines the portion of `N` that is used for training. The rest is used for testing.
11. `'file_format'` : determines the file format. Either `'tfrecords'` or `'hdf5'`.
12. `'compression'` : compression format for `tfrecords` dataset. Keep 'GZIP'.
13. `'name'` : the name used in `frecords` for the input of dataset. TODO: more details and features coming soon.

Please don't leave any of these fields empty even if they are not relevant to the characteristics you have in mind.




### For Developers
Coming soon (how to change the model structure, how to change the loss function, etc).

features

parameters

for developers

for users
