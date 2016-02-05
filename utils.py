#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/01/18 

import numpy as np

def pascal_classes():
  '''
  classes name and id table
  '''
  classes = {'aeroplane' : 1,  'bicycle'   : 2,  'bird'        : 3,  'boat'         : 4, 
             'bottle'    : 5,  'bus'       : 6,  'car'        : 7,  'cat'          : 8, 
             'chair'     : 9,  'cow'       : 10, 'diningtable' : 11, 'dog'          : 12, 
             'horse'     : 13, 'motorbike' : 14, 'person'      : 15, 'potted plant' : 16, 
             'sheep'     : 17, 'sofa'      : 18, 'train'       : 19, 'tv/monitor'   : 20}

  return classes

def pascal_palette():
  '''
  pascal voc rgb and classes id table
  '''
  palette = {(  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20 }

  return palette

def convert_from_color_segmentation(arr_3d):
  '''
  convert pascal voc bgr to gray scale as clases index

  arguments:
    arr_3d: rgb image
  '''
  arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
  palette = pascal_palette()

  # slow!
  for i in range(0, arr_3d.shape[0]):
    for j in range(0, arr_3d.shape[1]):
      # each channel's tuple is key of rgb and index table
      key = (arr_3d[i,j,0], arr_3d[i,j,1], arr_3d[i,j,2])
      arr_2d[i, j] = palette.get(key, 0) # default value if key was not found is 0

  return arr_2d

def get_id_classes(classes):
  '''
  get index of class name
  arguments:
    classes: array of class names
  '''
  all_classes = pascal_classes()
  id_classes = [all_classes[c] for c in classes]
  return id_classes

def strstr(str1, str2):
  '''
  if str1 include str2 as sub set, return True
  '''
  if str1.find(str2) != -1:
    return True
  else:
    return False


if __name__ == '__main__':
  print "['bus', 'car'] --->"
  print get_id_classes(['bus', 'car'])
