#!/bin/bash

spec=woof
net=resnet # resnet_ap resnet
d=10
ipc=10
path=../results/dit-distillation/woof-50-dit-igd-ckpts-convnet6-k5-gamma120-r1-gi200-low30-high45

python train.py -d imagenet --imagenet_dir ${path} ../imagenet/ -n ${net} --depth ${d} --nclass 1000 --norm_type instance --ipc ${ipc} --tag test --slct_type random --spec ${spec}

python IGD/train.py -d imagenet --imagenet_dir ../imagenet /data2/wlf/datasets/imagenet/ -n resnet --depth 10 --nclass 1000 --norm_type instance --ipc 10 --tag test --slct_type random --spec 1k --epoch_print_freq 1

python IGD/train.py -d imagenet --imagenet_dir imagenet/ipc_10 /data2/wlf/datasets/imagenet/ -n resnet --depth 10 --nclass 1000 --norm_type instance --ipc 10 --tag test --slct_type random --spec 1k --batch_size 128 --verbose


