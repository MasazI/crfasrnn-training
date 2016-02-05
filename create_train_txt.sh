#!/bin/bash

if [ $# -ne 1 ];then
    echo "Usage: create_train_txt.sh <labels image directory path>"
    exit
fi

labels_dir=$1
find ${labels_dir}/ -printf '%f\n' | sed 's/\.png//' | tail -n +2 > train.txt

