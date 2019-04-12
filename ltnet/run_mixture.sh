#!/usr/bin/env sh

/home/ubuntu/caffe/build/tools/caffe train \
--solver=solver.prototxt \
--gpu=0
#--snapshot=snapshot/train_fer_iter_100000.solverstate
