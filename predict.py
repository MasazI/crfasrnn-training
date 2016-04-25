# encoding: utf-8

import os
import os.path as osp
import sys

caffe_path = osp.join(os.getenv('CRF_AS_RNN_PATH'), 'caffe-crfrnn', 'python')
print("add path: %s" % (caffe_path))
sys.path.insert(0, caffe_path)

import cPickle
import logging
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import cStringIO as StringIO
import caffe
#import matplotlib.pyplot as plt

import cv2

image_size = 100

pallete = [0,0,0,
             128,0,0,
             0,128,0,
             128,128,0,
             0,0,128,
             128,0,128,
             0,128,128,
             128,128,128,
             64,0,0,
             192,0,0,
             64,128,0,
             192,128,0,
             64,0,128,
             192,0,128,
             64,128,128,
             192,128,128,
             0,64,0,
             128,64,0,
             0,192,0,
             128,192,0,
             0,64,128,
             128,64,128,
             0,192,128,
             128,192,128,
             64,64,0,
             192,64,0,
             64,192,0,
             192,192,0]


class CRFasRNN:
    '''
    crf as rnn class
    '''
    def __init__(self, model_file, trained_model):
        self.model_file = model_file
        self.trained_model = trained_model
        self.net = caffe.Segmenter(model_file, trained_model, gpu=True)

    def predict_image(self, mat, debug=True):
        '''
        arguments:
            image: opencv format image.
        return:
            labeld image
            ratio
        '''
        if debug:
            print mat.shape
            print mat

        input_image = mat[:,:,(2,1,0)]
        input_image = input_image.astype(np.float)
        if debug:
            print input_image.shape
            print input_image

        min_size = 400.
        original_width = input_image.shape[1]
        original_height = input_image.shape[0]
        ratio = .0
        if original_width > original_height:
            print("change scale with width")
            ratio = 400./original_width
        else:
            print("change scale with height")
            ratio = 400./original_height

        print("width: %f --> %f" % (original_width, original_width * ratio))
        print("height: %f --> %f" % (original_height, original_height * ratio))

        input_image = caffe.io.resize_image(input_image, (int(original_height * ratio), int(original_width * ratio)))
        
        width = input_image.shape[0]
        height = input_image.shape[1]
        maxDim = max(width, height)

        image = PILImage.fromarray(np.uint8(input_image))
        image = np.array(image)

        mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
        reshaped_mean_vec = mean_vec.reshape(1, 1, 3)

        im = image[:,:,::-1]
        im = im - reshaped_mean_vec

        cur_h, cur_w, cur_c = im.shape

        if cur_h >= 500 or cur_w >= 500:
            print("image is too big.")
            return

        pad_h = 500 - cur_h
        pad_w = 500 - cur_w

        im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

        segmentation = self.net.predict([im])
        segmentation2 = segmentation[0:cur_h, 0:cur_w]

        output_im = PILImage.fromarray(segmentation2)
        output_im.putpalette(pallete)

        mat_predicted = np.asarray(output_im)

        print mat_predicted.shape
        print mat_predicted

        if debug:
            #cv2.imwrite('output_cv_im.png', mat_predicted*255)
            cv2.imshow('labeled each pixcels.', mat_predicted*255)
            cv2.waitKey(0)
        return mat_predicted, ratio


    def predict(self, image_file_path):
        mat = cv2.imread(image_file_path)
        predicted, _ = self.predict_image(mat)
        

if __name__ == '__main__':
        argvs = sys.argv
        argc = len(argvs)
        if argc != 4:
            print("Usage: predict.py <deploy prototxt> <trained model> <image path>")
            exit()

        deploy_prototxt = argvs[1]
        trained_model = argvs[2]
        image_path = argvs[3]

        crf_as_rnn = CRFasRNN(deploy_prototxt, trained_model)
        crf_as_rnn.predict(image_path)
