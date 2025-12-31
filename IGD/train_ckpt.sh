#!/bin/bash
model=convnet6
spec=1k

python IGD/train_ckpts.py -d imagenet --imagenet_dir /data2/wlf/datasets/imagenet/train/ /data2/wlf/datasets/imagenet \
    -n ${model} --nclass 1000 --norm_type instance --tag test --slct_type random --spec ${spec} --mixup vanilla --repeat 1 \
    --ckpt-dir ./ckpts/${spec}/${model}/ --epochs 50 --depth 6 --ipc -1

python IGD/train_ckpts.py -d imagenet --imagenet_dir /data/wlf/datasets/imagenette2/train/ /data/wlf/datasets/imagenette2 -n resnet --nclass 10 --norm_type instance --tag test --slct_type random --spec nette --mixup vanilla --repeat 1 --ckpt-dir ./ckpts/nette/resnet10/ --epochs 50 --depth 10 --ipc -1