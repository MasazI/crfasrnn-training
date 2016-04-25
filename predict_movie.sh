#!/bin/sh

if [ $# -ne 1 ]; then
    echo "Usage: predict_movie.sh <movie file path>"
    exit 1
fi

MOVIE_FILE_PATH=$1
echo ${MOVIE_FILE_PATH}

export CAFFE_HOME=$HOME/caffe
export PYTHONPATH=$HOME/caffe/python:/usr/local/lib/python2.7/site-packages:$PYTHONPATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:/usr/tools/lib:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda-7.0
export CUDA_LAUNCH_BLOCKING=1
export CUDA_INC_DIR=/usr/local/cuda/bin:$CUDA_INC_DIR
# for crf as rnn
export CRF_AS_RNN_PATH=/home/deep/crfasrnn

python predict_movie.py ${MOVIE_FILE_PATH}
