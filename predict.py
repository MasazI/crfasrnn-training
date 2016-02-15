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
import matplotlib.pyplot as plt

image_size = 500


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

    def predict(self, image_file_path):
        input_image = 255 * caffe.io.load_image(image_file_path)

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
            return

        pad_h = 500 - cur_h
        pad_w = 500 - cur_w

        im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

        segmentation = self.net.predict([im])
        segmentation2 = segmentation[0:cur_h, 0:cur_w]

        output_im = PILImage.fromarray(segmentation2)
        output_im.putpalette(pallete)

        plt.imshow(output_im)
        plt.savefig('output.png')

if __name__ == '__main__':
        crf_as_rnn = CRFasRNN('TVG_CRFRNN_COCO_VOC.prototxt', 'models/train_iter_29000.caffemodel')
        crf_as_rnn.predict('test_images/car.jpg')



