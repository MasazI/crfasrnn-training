import argparse

import os
import sys
import os.path as osp


import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe

import time

caffe_path = osp.join(os.getenv('CRF_AS_RNN_PATH'), 'caffe-crfrnn', 'python')
print("add path: %s" % (caffe_path))
sys.path.insert(0, caffe_path)

from predict import CRFasRNN

import cv2

import cPickle
import logging
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description='Evaluate a Caffe reference model on ILSVRC2012 dataset')
parser.add_argument('movie', help='Path to movie file')
parser.add_argument('--skip', '-s', type=int, default=0, help='The number of skip frames.')
args = parser.parse_args()


# crf as rnn object
crfasrnn = CRFasRNN('predict.prototxt', 'TVG_CRFRNN_COCO_VOC.caffemodel')

# get video capture
cap = cv2.VideoCapture(args.movie)
cnt = args.skip

print cap.isOpened()

#cv2.imshow('predicted Each Pixcel Demo.')

output_cnt = 1
while(cap.isOpened()):
    ret, frame = cap.read()

    height = frame.shape[0]
    width = frame.shape[1]

    print("height %d" % (height))
    print("width %d" % (width))

    if frame is None:
        cnt = cnt + 1
        cap.set(1, cnt)
        print 'stop'
        exit()

    output_cnt_filename = "%04d" % (output_cnt) + ".jpg"
    mat_predicted, ratio = crfasrnn.predict_image(frame, debug=False)
    mat_predicted_orgsize = cv2.resize(mat_predicted*255, (width, height))
    cv2.imshow("Original Movie.", frame)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "original_movie_frame", output_cnt_filename), frame)
    cv2.imshow('Predicted Each Pixcel Demo.', mat_predicted_orgsize)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "predict_frame", output_cnt_filename), mat_predicted_orgsize)

    output_cnt += 1

    cv2.waitKey(20)

    cnt = cnt + 5
    # CV_CAP_PROP_POS_FRAMES is 1   
    cap.set(1, cnt)


cap.release()
cv2.destroyAllWindows()

