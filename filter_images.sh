#!/bin/sh
if [ $# -ne 2 ];then
    echo "Usage: filter_images.sh <labels image directory path> <labels image list path>"
    exit
fi

labels_dir=$1
labels_list=$2

python filter_images.py ${labels_dir}/ ${labels_list}
