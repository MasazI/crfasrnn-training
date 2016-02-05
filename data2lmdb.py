#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/01/18

from __future__ import print_function
import os
import sys
import lmdb
from random import shuffle
from skimage.io import imread
from scipy.misc import imresize
import numpy as np
from PIL import Image
import caffe
from utils import get_id_classes, convert_from_color_segmentation

# preprecess mode
preprocess_mode = 'pad'

# dimention of input layer
im_sz = 500
class_names = ['car', 'motorbike', 'bus']

# ratio of test set in all data set
test_ratio = 0.1

# image extension
image_ext = '.jpg'

# labelled segmentation image extension
label_ext = '.png'

def main():
  labels_path = process_arguments(sys.argv)
  class_ids = get_id_classes(class_names)
  train_imgs, test_imgs = split_train_test_imgs(class_names, test_ratio)

  ## Train
  # Images
  print('Train images')
  path_src = 'images/'
  path_dst = 'train_images_3_lmdb'
  convert2lmdb(path_src, train_imgs, image_ext, path_dst, class_ids, preprocess_mode, im_sz, 'image')

  # Labels
  print('Train labels')
  if labels_path:
    path_src = labels_path
  else:
    path_src = 'labels/'

  path_dst = 'train_labels_3_lmdb'
  convert2lmdb(path_src, train_imgs, label_ext, path_dst, class_ids, preprocess_mode, im_sz, 'label')

  ## Test
  # Images
  print('Test images')
  path_src = 'images/'
  path_dst = 'test_images_3_lmdb'
  convert2lmdb(path_src, test_imgs, image_ext, path_dst, class_ids, preprocess_mode, im_sz, 'image')

  # Labels
  print('Test labels')
  if labels_path:
    path_src = labels_path
  else:
    path_src = 'labels/'
  path_dst = 'test_labels_3_lmdb'
  convert2lmdb(path_src, test_imgs, label_ext, path_dst, class_ids, preprocess_mode, im_sz, 'label')

def split_train_test_imgs(class_names, test_ratio):
  train_imgs = []
  test_imgs = []

  for i in class_names:
    file_name = i + '.txt' 
    num_lines = get_num_lines(file_name)
    num_test_imgs = test_ratio * num_lines
    current_line = 1

    with open(file_name, 'rb') as f:
      for line in f:
        if current_line < num_test_imgs:
          test_imgs.append(line.strip())
        else:
          train_imgs.append(line.strip())

        current_line += 1

  shuffle(train_imgs)
  shuffle(test_imgs)

  print(str(len(train_imgs)) + ' train images')
  print(str(len(test_imgs)) + ' test images')

  return train_imgs, test_imgs

def get_num_lines(file_name):
  num_lines = 0

  with open(file_name, 'rb') as f:
    for line in f:
      num_lines += 1

  return num_lines

# DEPRECATED
def get_src_imgs(file_name, ext):
  src_imgs = []

  with open(file_name, 'rb') as f:
      for img_file in f:
        img_file = img_file.strip()
        src_imgs.append(img_file + ext)

  return src_imgs

def convert2lmdb(path_src, src_imgs, ext, path_dst, class_ids, preprocess_mode, im_sz, data_mode):
  if os.path.isdir(path_dst):
    print('DB ' + path_dst + ' already exists.\n'
          'Skip creating ' + path_dst + '.', file=sys.stderr)
    return None

  if data_mode == 'label':
    lut = create_lut(class_ids)

  db = lmdb.open(path_dst, map_size=int(1e12))

  with db.begin(write=True) as in_txn:
    for idx, img_name in enumerate(src_imgs):
      #img = imread(os.path.join(path_src + img_name)+ext)
      img = np.array(Image.open(os.path.join(path_src + img_name)+ext))
      img = img.astype(np.uint8)

      if data_mode == 'label':
        img = preprocess_label(img, lut, preprocess_mode, im_sz)
      elif data_mode == 'image':
        img = preprocess_image(img, preprocess_mode, im_sz)

      img_dat = caffe.io.array_to_datum(img)
      in_txn.put('{:0>10d}'.format(idx), img_dat.SerializeToString())

def preprocess_image(img, mode, im_sz):
  img = preprocess_data(img, mode, im_sz, 'image')
  img = img[:,:,::-1] # RGB to BGR
  img = img.transpose((2,0,1))
  return img

def preprocess_label(img, lut, mode, im_sz):
  # If label is three-dimensional image we have to convert it to
  # corresponding labels (0 - 20). Currently anticipated labels are from
  # VOC pascal datasets.
  if (len(img.shape) > 2):
    img = convert_from_color_segmentation(img)

  img = preprocess_data(img, mode, im_sz, 'label')
  img = lut[img]
  img = np.expand_dims(img, axis=0)
  #img = _2D_to_ND(img, len(np.unique(lut)))
  #img = img.transpose((2,0,1))
  return img

def create_lut(class_ids, max_id=256):
  lut = np.zeros(max_id, dtype=np.uint8)

  new_index = 1
  for i in class_ids:
    lut[i] = new_index
    new_index += 1

  return lut

def _2D_to_ND(label, n_levels):
  nd_label = np.zeros((label.shape[0], label.shape[1], n_levels)).astype(np.uint8)
  for l in range(n_levels):
    nd_label[:,:,l] = (label==l) * 1

  return nd_label

def preprocess_data(img, preprocess_mode, im_sz, data_mode):
  if preprocess_mode == 'pad':

    if data_mode == 'image':
      img = np.pad(img, ((0, im_sz-img.shape[0]), (0, im_sz-img.shape[1]), (0,0)), 'constant', constant_values=(0))
    elif data_mode == 'label':
      img = np.pad(img, ((0, im_sz-img.shape[0]), (0, im_sz-img.shape[1])), 'constant', constant_values=(0))
    else:
      print('Invalid data mode.', file=sys.stderr)

  elif preprocess_mode == 'res':
    img = imresize(img, (im_sz, im_sz), interp='bilinear')
  else:
    print('Invalid preprocess mode.', file=sys.stderr)

  return img

def process_arguments(argv):
  new_path = None

  if len(argv) > 2:
    help()
  elif len(argv) == 2:
    new_path = argv[1]

  return new_path

def help():
  print('Usage: python data2lmdb.py [PATH]\n'
        'PATH points to a directory with ground truth segmentation images.'
        , file=sys.stderr)

  exit()
    
if __name__ == '__main__':
  main()
