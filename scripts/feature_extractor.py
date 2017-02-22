import json
import traceback

import cv2

__author__ = 'ananya'

import caffe
import sys

sys.path.append("..")
from scipy.misc import imresize
import numpy as np
import os
from datetime import datetime


class FeatureExtractor(object):
    def __init__(self, path_to_deploy_file, path_to_model_file, input_layer_name="data_q", gpu_mode=True, device_id=1,
                 height=None, width=None):
        self.path_to_deploy_file = path_to_deploy_file
        self.path_to_model_file = path_to_model_file
        if gpu_mode:
            caffe.set_mode_gpu()
            caffe.set_device(device_id)
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Net(path_to_deploy_file, path_to_model_file, caffe.TEST)
        self.input_layer_name = input_layer_name
        self.height = height or self.net.blobs[self.input_layer_name].data.shape[2]
        self.width = width or self.net.blobs[self.input_layer_name].data.shape[3]

    def extract_one(self, img_path, layer):
        img = self.getImageFromPath(img_path)
        resized_img = imresize(img, (self.height, self.width), 'bilinear')
        transposed_img = np.transpose(resized_img, (2, 0, 1))
        assert self.net.blobs[self.input_layer_name].data.shape == (1,) + transposed_img.shape
        self.net.blobs[self.input_layer_name].data[...] = transposed_img
        self.net.forward()
        fv = self.net.blobs[layer].data[0].flatten()
        return fv

    def extract_batch(self, img_paths, layer):
        batch_size = len(img_paths)
        fv_dict = {}
        start_time = datetime.now()
        resized_imgs = []
        for path in img_paths:
            try:
                img = self.getImageFromPath(path)
                resized_imgs.append(imresize(img, (self.height, self.width), 'bilinear'))
            except Exception as e:
                print "Exception for image", path
                traceback.print_exc()

        transposed_imgs = [np.transpose(x, (2, 0, 1)) for x in resized_imgs]
        reqd_shape = (batch_size,) + transposed_imgs[0].shape
        self.net.blobs[self.input_layer_name].reshape(*reqd_shape)
        self.net.blobs[self.input_layer_name].data[...] = transposed_imgs
        self.net.forward()
        fv = self.net.blobs[layer].data
        count = 0
        for img_path in img_paths:
            fv_key = os.path.splitext(os.path.basename(img_path))[0]
            fv_value = fv[count].flatten()
            fv_dict[fv_key] = fv_value
            count += 1
        end_time = datetime.now()
        delta = end_time - start_time
        print("Batch took " + str(delta.total_seconds() * 1000))
        return fv_dict

    def getImageFromPath(self, path):
        return cv2.imread(path, cv2.IMREAD_COLOR)
