#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:09:42 2019
@author: M. Hossein Eybposh
Module for hologram generation algorithms. Currently (Oct. 2019), the only only
algorithm that is available is the tensorflow implementation of Gerchberg–Saxton
algorithm. Other algorithms like the HoloNet wil be added in the future.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from queue import Queue
from threading import Thread
import warnings
from skimage.draw import circle, line_aa
import numpy as np
from tqdm import tqdm
import h5py as h5
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Input, Concatenate, Lambda
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D

#%%
class DeepCGH_Datasets(object):
    '''
    Class for the Dataset object used in DeepCGH algorithm.
    Inputs:
        path   string, determines the lcoation that the datasets are going to be stored in
        num_iter   int, determines the number of iterations of the GS algorithm
        input_shape   tuple of shape (height, width)
    Returns:
        Instance of the object
    '''
    def __init__(self, params):
        try:
            assert params['object_type'] in ['Disk', 'Line', 'Dot'], 'Object type not supported'
            
            self.path = params['path']
            self.shape = params['shape']
            self.N = params['N']
            self.ratio = params['train_ratio']
            self.object_size = params['object_size']
            self.intensity = params['intensity']
            self.object_count = params['object_count']
            self.name = params['name']
            self.object_type = params['object_type']
            self.centralized = params['centralized']
            self.normalize = params['normalize']
            self.compression = params['compression']
        except:
            assert False, 'Not all parameters are provided!'
            
        self.__check_avalability()
        
    
    def __check_avalability(self):
        print('Current working directory is:')
        print(os.getcwd(),'\n')
        
        self.filename = self.object_type + '_SHP{}_N{}_SZ{}_INT{}_Crowd{}_CNT{}_Split.tfrecords'.format(self.shape, 
                                           self.N, 
                                           self.object_size,
                                           self.intensity, 
                                           self.object_count,
                                           self.centralized)
        
        self.absolute_file_path = os.path.join(os.getcwd(), self.path, self.filename)
        if not (os.path.exists(self.absolute_file_path.replace('Split', '')) or os.path.exists(self.absolute_file_path.replace('Split', 'Train')) or os.path.exists(self.absolute_file_path.replace('Split', 'Test'))):
            warnings.warn('File does not exist. New dataset will be generated once getDataset is called.')
            print(self.absolute_file_path)
        else:
            print('Data already exists.')
           
            
    def __get_line(self, shape, start, end):
        img = np.zeros(shape, dtype=np.float32)
        rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
        img[rr, cc] = val * 1
        return img
    
    
    def get_circle(self, shape, radius, location):
        """Creates a single circle.
    
        Parameters
        ----------
        shape : tuple of ints
            Shape of the output image
        radius : int
            Radius of the circle.
        location : tuple of ints
            location (x,y) in the image
    
        Returns
        -------
        img
            a binary 2D image with a circle inside
        rr2, cc2
            the indices for a circle twice the size of the circle. This is will determine where we should not create circles
        """
        img = np.zeros(shape, dtype=np.float32)
        rr, cc = circle(location[0], location[1], radius, shape=img.shape)
        img[rr, cc] = 1
        # get the indices that are forbidden and return it
        rr2, cc2 = circle(location[0], location[1], 2*radius, shape=img.shape)
        return img, rr2, cc2


    def __get_allowables(self, allow_x, allow_y, forbid_x, forbid_y):
        '''
        Remove the coords in forbid_x and forbid_y from the sets of points in
        allow_x and allow_y.
        '''
        for i in forbid_x:
            try:
                allow_x.remove(i)
            except:
                continue
        for i in forbid_y:
            try:
                allow_y.remove(i)
            except:
                continue
        return allow_x, allow_y
    
    
    def __get_randomCenter(self, allow_x, allow_y):
        list_x = list(allow_x)
        list_y = list(allow_y)
        ind_x = np.random.randint(0,len(list_x))
        ind_y = np.random.randint(0,len(list_y))
        return list_x[ind_x], list_y[ind_y]
    
    
    def __get_randomStartEnd(self, shape):
        start = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
        end = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
        return start, end


    #% there shouldn't be any overlap between the two circles 
    def __get_RandDots(self, shape, maxnum = [10, 20]):
        '''
        returns a single sample (2D image) with random dots
        '''
        # number of random lines
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        xs = list(np.random.randint(0, shape[0], (n,)))
        ys = list(np.random.randint(0, shape[1], (n,)))
        
        for x, y in zip(xs, ys):
            image[x, y] = 1
            
        return image

    #% there shouldn't be any overlap between the two circles 
    def __get_RandLines(self, shape, maxnum = [10, 20]):
        '''
        returns a single sample (2D image) with random lines
        '''
        # number of random lines
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        for i in range(n):
            # generate centers
            start, end = self.__get_randomStartEnd(shape)
            
            # get circle
            img = self.__get_line(shape, start, end)
            image += img
        image -= image.min()
        image /= image.max()
        return image
    
    #% there shouldn't be any overlap between the two circles 
    def __get_RandBlobs(self, shape, maxnum = [10,12], radius = 5, intensity = 1):
        '''
        returns a single sample (2D image) with random blobs
        '''
        # random number of blobs to be generated
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        try: # in case the radius of the blobs is variable, get the largest diameter
            r = radius[-1]
        except:
            r = radius
        
        # define sets for storing the values
        allow_x = set(range(shape[0]))
        allow_y = set(range(shape[1]))
        if not self.centralized:
            forbid_x = set(list(range(r)) + list(range(shape[0]-r, shape[0])))
            forbid_y = set(list(range(r)) + list(range(shape[1]-r, shape[1])))
        else:
            forbid_x = set(list(range(r)) + list(range(shape[0]-r, shape[0])) + list(range(shape[0]//6, (5)*shape[0]//6)))
            forbid_y = set(list(range(r)) + list(range(shape[1]-r, shape[1])) + list(range(shape[1]//6, (5)*shape[1]//6)))
        
        allow_x, allow_y = self.__get_allowables(allow_x, allow_y, forbid_x, forbid_y)
        count = 0
        # else
        for i in range(n):
            # generate centers
            x, y = self.__get_randomCenter(allow_x, allow_y)
            
            if isinstance(radius, list):
                r = int(np.random.randint(radius[0], radius[1]))
            else:
                r = radius
            
            if isinstance(intensity, list):
                int_4_this = int(np.random.randint(np.round(intensity[0]*100), np.round(intensity[1]*100)))
                int_4_this /= 100.
            else:
                int_4_this = intensity
            
            # get circle
            img, xs, ys = self.get_circle(shape, r, (x,y))
            allow_x, allow_y = self.__get_allowables(allow_x, allow_y, set(xs), set(ys))
            image += img * int_4_this
            count += 1
            if len(allow_x) == 0 or len(allow_y) == 0:
                break
        return image
    
    
    def coord2image(self, coords):
        num_planes = self.shape[-1]
        
        sample = np.zeros(self.shape)
        
        for plane in range(num_planes):
            canvas = np.zeros(self.shape[:-1], dtype=np.float32)
        
            for i in range(coords.shape[-1]):
                img, _, __ = self.get_circle(self.shape[:-1], self.object_size, [coords[0, i], coords[1, i]])
                canvas += img.astype(np.float32)
            
            sample[:, :, plane] = (canvas>0)*1.
            
            if (num_planes > 1) and (plane != 0 and self.normalize == True):
                sample[:, :, plane] *= np.sqrt(np.sum(sample[:, :, 0]**2)/np.sum(sample[:, :, plane]**2))
            
        sample -= sample.min()
        sample /= sample.max()
        
        return np.expand_dims(sample, axis = 0)
                
    
    def get_randSample(self):
        
        num_planes = self.shape[-1]
        
        sample = np.zeros(self.shape)
        
        for plane in range(num_planes):
            if self.object_type == 'Disk':
                img = self.__get_RandBlobs(shape = (self.shape[0], self.shape[1]),
                                           maxnum = self.object_count,
                                           radius = self.object_size,
                                           intensity = self.intensity)
            elif self.object_type == 'Line':
                img = self.__get_RandLines((self.shape[0], self.shape[1]),
                                           maxnum = self.object_count)
            elif self.object_type == 'Dot':
                img = self.__get_RandDots(shape = (self.shape[0], self.shape[1]),
                                          maxnum = self.object_count)
                

            sample[:, :, plane] = img
            
            if (num_planes > 1) and (plane != 0 and self.normalize == True):
                sample[:, :, plane] *= np.sqrt(np.sum(sample[:, :, 0]**2)/np.sum(sample[:, :, plane]**2))
        
        sample -= sample.min()
        sample /= sample.max()
        
        return sample
    
    
    def __bytes_feature(self, value):
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
    
    
    def __int64_feature(self, value):
      return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
    
    
    def __generate(self):
        '''
        Creates a dataset of randomly located blobs and stores the data in an TFRecords file. Each sample (3D image) contains
        a randomly determined number of blobs that are randomly located in individual planes.
        Inputs:
            filename : str
                path to the dataset file
            N: int
                determines the number of samples in the dataset
            fraction : float
                determines the fraction of N that is used as "train". The rest will be the "test" data
            shape: (int, int)
                tuple of integers, shape of the 2D planes
            maxnum: int
                determines the max number of blobs
            radius: int
                determines the radius of the blobs
            intensity : float or [float, float]
                intensity of the blobs. If a scalar, it's a binary blob. If a list, first element is the min intensity and
                second one os the max intensity.
            normalize : bool
                flag that determines whether the 3D data is normalized for fixed energy from plane to plane
    
        Outputs:
            aa:
    
            out_dataset:
                numpy.ndarray. Numpy array with shape (samples, x, y)
        '''
        
#        assert self.shape[-1] > 1, 'Wrong dimensions {}. Number of planes cannot be {}'.format(self.shape, self.shape[-1])
        
        train_size = np.floor(self.ratio * self.N)
        # TODO multiple tfrecords files to store data on. E.g. every 1000 samples in one file
        options = tf.io.TFRecordOptions(compression_type = self.compression)
#        options = None
        with tf.io.TFRecordWriter(self.absolute_file_path.replace('Split', 'Train'), options = options) as writer_train:
            with tf.io.TFRecordWriter(self.absolute_file_path.replace('Split', 'Test'), options = options) as writer_test:
                for i in tqdm(range(self.N)):
                    sample = self.get_randSample()
                    
                    image_raw = sample.tostring()
                    
                    feature = {'sample': self.__bytes_feature(image_raw)}
                    
                    # 2. Create a tf.train.Features
                    features = tf.train.Features(feature = feature)
                    # 3. Createan example protocol
                    example = tf.train.Example(features = features)
                    # 4. Serialize the Example to string
                    example_to_string = example.SerializeToString()
                    # 5. Write to TFRecord
                    if i < train_size:
                        writer_train.write(example_to_string)
                    else:
                        writer_test.write(example_to_string)
            
    
    def getDataset(self):
        if not (os.path.exists(self.absolute_file_path.replace('Split', '')) or os.path.exists(self.absolute_file_path.replace('Split', 'Train')) or os.path.exists(self.absolute_file_path.replace('Split', 'Test'))):
            print('Generating data...')
            folder = os.path.join(os.getcwd(), self.path)
            
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            self.__generate()
        self.dataset_paths = [self.absolute_file_path.replace('Split', 'Train'), self.absolute_file_path.replace('Split', 'Test')]
        
     

#%%
class DeepCGH(object):
    '''
    Class for the DeepCGH algorithm.
    Inputs:
        batch_size   int, determines the batch size of the prediction
        num_iter   int, determines the number of iterations of the GS algorithm
        input_shape   tuple of shape (height, width)
    Returns:
        Instance of the object
    '''
    def __init__(self,
                 data_params,
                 model_params):
        
        
        self.path = model_params['path']
        self.shape = data_params['shape']
        self.plane_distance = model_params['plane_distance']
        self.n_kernels = model_params['n_kernels']
        self.IF = model_params['int_factor']
        self.wavelength = model_params['wavelength']
        self.ps = model_params['pixel_size']
        self.object_type = data_params['object_type']
        self.centralized = data_params['centralized']
        self.input_name = model_params['input_name']
        self.output_name = model_params['output_name']
        self.token = model_params['token']
        self.zs = [-1*self.plane_distance*x for x in np.arange(1, (self.shape[-1]-1)//2+1)][::-1] + [self.plane_distance*x for x in np.arange(1, (self.shape[-1]-1)//2+1)]
        self.input_queue = Queue(maxsize=4)
        self.output_queue = Queue(maxsize=4)
        self.__check_avalability()
        self.lr = model_params['lr']
        self.batch_size = model_params['batch_size']
        self.epochs = model_params['epochs']
        self.token = model_params['token']
        self.shuffle = model_params['shuffle']
        self.max_steps = model_params['max_steps']
        
        
    def __start_thread(self):
        self.prediction_thread = Thread(target=self.__predict_from_queue, daemon=True)
        self.prediction_thread.start()
        
        
    def __check_avalability(self):
        print('Looking for trained models in:')
        print(os.getcwd(), '\n')
        
        self.filename = 'Model_{}_SHP{}_IF{}_Dst{}_WL{}_PS{}_CNT{}_{}'.format(self.object_type,
                                                                              self.shape, 
                                                                              self.IF,
                                                                              self.plane_distance,
                                                                              self.wavelength,
                                                                              self.ps,
                                                                              self.centralized,
                                                                              self.token)
        
        self.absolute_file_path = os.path.join(os.getcwd(), self.path, self.filename)
        
        if not os.path.exists(self.absolute_file_path):
            print('No trained models found. Please call the `train` method. \nModel checkpoints will be stored in: \n {}'.format(self.absolute_file_path))
            
        else:
            print('Model already exists.')
    
    
    def __make_folders(self):
        if not os.path.exists(self.absolute_file_path):
            os.makedirs(self.absolute_file_path)
    
        
    def train(self, deepcgh_dataset, lr = None, batch_size = None, epochs = None, token = None, shuffle = None, max_steps = None):
        # Using default params or new ones?
        if lr is None:
            lr = self.lr
        if batch_size is None:
            batch_size = self.batch_size
        if epochs is None:
            epochs = self.epochs
        if token is None:
            token = self.token
        if shuffle is None:
            shuffle = self.shuffle
        if max_steps is None:
            max_steps = self.max_steps
        
        # deifne Estimator
        model_fn = self.__get_model_fn()
        
        # Data
        train, validation = self.load_data(deepcgh_dataset.dataset_paths, batch_size, epochs, shuffle)
        
        self.__make_folders()
            
        self.estimator = tf.estimator.Estimator(model_fn,
                                                model_dir=self.absolute_file_path)
        
        # 
        train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=max_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=validation)
        
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

        self.__start_thread()
            
    
    def load_data(self, path, batch_size, epochs, shuffle):
        if isinstance(path, list) and ('tfrecords' in path[0]) and ('tfrecords' in path[1]):
            image_feature_description = {'sample': tf.io.FixedLenFeature([], tf.string)}
            
            def __parse_image_function(example_proto):
                parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
                img = tf.cast(tf.reshape(tf.io.decode_raw(parsed_features['sample'], tf.float64), self.shape), tf.float32)
                return {'target':img}, {'recon':img}
            
            def val_func():
                validation = tf.data.TFRecordDataset(path[1],
                                      compression_type='GZIP',
                                      buffer_size=None,
                                      num_parallel_reads=2).map(__parse_image_function).batch(batch_size)#.prefetch(tf.data.experimental.AUTOTUNE)
                
                return validation
            
            def train_func():
                train = tf.data.TFRecordDataset(path[0],
                                      compression_type='GZIP',
                                      buffer_size=None,
                                      num_parallel_reads=2).map(__parse_image_function).repeat(epochs).shuffle(shuffle).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
                return train
            #.repeat(epochs)
            return train_func, val_func
        else:
            raise('You got a problem in your file name bruh')
    
        
    def __generate_from_queue(self):
        '''
        A generator with infinite loop to fetch one smaple from the queue
        Returns:
            one sample!
        '''
        while True:
            yield self.input_queue.get()


    def __predict_from_queue(self):
        '''
        Once the input queue has something to offer to the estimator, the
        estimator makes the predictions and the outputs of that prediction are
        fed into the output queue.
        '''
        for i in self.estimator.predict(input_fn=self.__queued_predict_input_fn,
                                        yield_single_examples=False):
            self.output_queue.put(i)
        
    
    def get_hologram(self, inputs):
        '''
        Return the hologram using the GS algorithm with num_iter iterations.
        Inputs:
            inputs   numpy ndarray, the two dimentional target image
        Returns:
            hologram as a numpy ndarray 
        '''
        features = {}
        if not isinstance(self.input_name, str):
            for key, val in zip(self.input_name, inputs):
                features[key] = val
        else:
            features = {self.input_name: inputs}
        self.input_queue.put(features)
        predictions = self.output_queue.get()
        
        return predictions#[self.output_name]

    def __queued_predict_input_fn(self):
        '''
        Input function that returns a tensorflow Dataset from a generator.
        Returns:
            a tensorflow dataset
        '''
        # Fetch the inputs from the input queue
        type_dict = {}
        shape_dict = {}
        
        if not isinstance(self.input_name, str):
            for key in self.input_name:
                type_dict[key] = tf.float32
                shape_dict[key] = (None,)+self.shape
        else:
            type_dict = {self.input_name: tf.float32}
            shape_dict = {self.input_name:(None,)+self.shape}
        
        dataset = tf.data.Dataset.from_generator(self.__generate_from_queue,
                                                 output_types=type_dict,
                                                 output_shapes=shape_dict)
        return dataset
    
    
    def __get_model_fn(self):
        
        def interleave(x):
            return tf.nn.space_to_depth(input = x,
                                       block_size = self.IF,
                                       data_format = 'NHWC')
        
        
        def deinterleave(x):
            return tf.nn.depth_to_space(input = x,
                                       block_size = self.IF,
                                       data_format = 'NHWC')
        
        
        def __cbn(ten, n_kernels, act_func):
            x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(ten)
            x1 = BatchNormalization()(x1)
            x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(x1)
            x1 = BatchNormalization()(x1)
            return x1 
        
        
        def __cc(ten, n_kernels, act_func):
            x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(ten)
            x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(x1)
            return x1
        
        
        def __get_H(zs, shape, lambda_, ps):
            Hs = []
            for z in zs:
                x, y = np.meshgrid(np.linspace(-shape[1]//2+1, shape[1]//2, shape[1]),
                                   np.linspace(-shape[0]//2+1, shape[0]//2, shape[0]))
                fx = x/ps/shape[0]
                fy = y/ps/shape[1]
                exp = np.exp(-1j * np.pi * lambda_ * z * (fx**2 + fy**2))
                Hs.append(exp.astype(np.complex64))
            return Hs
        
        
        def __unet():
            n_kernels = self.n_kernels
            inp = Input(shape=self.shape, name='target')
            act_func = 'relu'
            x1_1 = Lambda(interleave, name='Interleave')(inp)
            # Block 1
            x1 = __cbn(x1_1, n_kernels[0], act_func)
            x2 = MaxPooling2D((2, 2), padding='same')(x1)
            # Block 2
            x2 = __cbn(x2, n_kernels[1], act_func)
            encoded = MaxPooling2D((2, 2), padding='same')(x2)
            # Bottleneck
            encoded = __cc(encoded, n_kernels[2], act_func)
            #
            x3 = UpSampling2D(2)(encoded)
            x3 = Concatenate()([x3, x2])
            x3 = __cc(x3, n_kernels[1], act_func)
            #
            x4 = UpSampling2D(2)(x3)
            x4 = Concatenate()([x4, x1])
            x4 = __cc(x4, n_kernels[0], act_func)
            #
            x4 = __cc(x4, n_kernels[1], act_func)
            x4 = Concatenate()([x4, x1_1])
            #
            phi_0_ = Conv2D(self.IF**2, (3, 3), activation=None, padding='same')(x4)
            phi_0 = Lambda(deinterleave, name='phi_0')(phi_0_)
            amp_0_ = Conv2D(self.IF**2, (3, 3), activation='relu', padding='same')(x4)
            amp_0 = Lambda(deinterleave, name='amp_0')(amp_0_)
            
            phi_slm = Lambda(__ifft_AmPh, name='phi_slm')([amp_0, phi_0])
            
            return Model(inp, phi_slm)
            
        
        def __accuracy(y_true, y_pred):
            denom = tf.sqrt(tf.reduce_sum(tf.pow(y_pred, 2), axis=[1, 2, 3])*tf.reduce_sum(tf.pow(y_true, 2), axis=[1, 2, 3]))
            return 1-tf.reduce_mean((tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])+1)/(denom+1), axis = 0)
        
        
        def __ifft_AmPh(x):
            '''
            Input is Amp x[1] and Phase x[0]. Spits out the angle of ifft.
            '''
            img = tf.dtypes.complex(tf.squeeze(x[0], axis=-1), 0.) * tf.math.exp(tf.dtypes.complex(0., tf.squeeze(x[1], axis=-1)))
            img = tf.signal.ifftshift(img, axes = [1, 2])
            fft = tf.signal.ifft2d(img)
            phase = tf.expand_dims(tf.math.angle(fft), axis=-1)
            return phase
        
        
        def __prop__(cf_slm, H = None, center = False):
            if not center:
                H = tf.broadcast_to(tf.expand_dims(H, axis=0), tf.shape(cf_slm))
                cf_slm *= tf.signal.fftshift(H, axes = [1, 2])
            fft = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(cf_slm, axes = [1, 2])), axes = [1, 2])
            img = tf.cast(tf.expand_dims(tf.abs(tf.pow(fft, 2)), axis=-1), dtype=tf.dtypes.float32)
            return img
        
        
        def __phi_slm(phi_slm):
            i_phi_slm = tf.dtypes.complex(np.float32(0.), tf.squeeze(phi_slm, axis=-1))
            return tf.math.exp(i_phi_slm)
        
        
        Hs = __get_H(self.zs, self.shape, self.wavelength, self.ps)
        
        
        def __big_loss(y_true, phi_slm):
            frames = []
            cf_slm = __phi_slm(phi_slm)
            for H, z in zip(Hs, self.zs):
                frames.append(__prop__(cf_slm, tf.keras.backend.constant(H, dtype = tf.complex64)))
            
            frames.insert(self.shape[-1] // 2, __prop__(cf_slm, center = True))
            
            y_pred = tf.concat(values=frames, axis = -1)
            
            return __accuracy(y_true, y_pred)
            
        
        def model_fn(features, labels, mode):
            unet = __unet()
            
            training = (mode == tf.estimator.ModeKeys.TRAIN)
            
            phi_slm = unet(features['target'], training = training)
        
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode, predictions = phi_slm)
            
            else:
                acc = __big_loss(labels['recon'], phi_slm)
                
                if mode == tf.estimator.ModeKeys.EVAL:
                    return tf.estimator.EstimatorSpec(mode, loss = acc)
                
                elif mode == tf.estimator.ModeKeys.TRAIN:
                    train_op = None
                    
                    opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
                    
                    opt.iterations = tf.compat.v1.train.get_or_create_global_step()
                    
                    update_ops = unet.get_updates_for(None) + unet.get_updates_for(features['target'])
                    
                    minimize_op = opt.get_updates(acc , unet.trainable_variables)[0]
                    
                    train_op = tf.group(minimize_op, *update_ops)
                    
                    return tf.estimator.EstimatorSpec(mode = mode,
                                                      predictions = {self.output_name: phi_slm},
                                                      loss = acc,
                                                      train_op = train_op)
        return model_fn


