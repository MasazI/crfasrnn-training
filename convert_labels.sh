#/bin/sh
if [ $# -ne 3 ];then
    echo "Usage: convert_labels.sh <labels image directory path> <labels image list path> <output converted image directory path>"
    exit
fi

labels_dir=$1
labels_list=$2
output_dir=$3

python convert_labels.py ${labels_dir}/ ${labels_list} ${output_dir}/
