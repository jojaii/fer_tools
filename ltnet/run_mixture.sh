#!/usr/bin/env sh

/home/ubuntu/caffe/build/tools/caffe train \
--solver=solver.prototxt \
--snapshot=snapshot/train_fer_iter_100000.solverstate
--gpu=0
