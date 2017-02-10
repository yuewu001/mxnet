#! /bin/bash
#################################################################################
#     File Name           :     run.sh
#     Created By          :     yuewu
#     Creation Date       :     [2017-02-09 16:31]
#     Last Modified       :     [2017-02-09 20:49]
#     Description         :      
#################################################################################


for layer in nw1_1 nw1_2 nw2_1 nw2_2 nw3_1 nw3_2 nw3_3 nw4_1 nw4_2 nw4_3 nw5_1 nw5_2 nw5_3
do
    for val in 1e-3 1e-2 1e-1 1
    do
        python train.py \
            --pretrain models/pretrain/cifar10 --load-epoch 200 \
            --model-prefix models/finetune/cifar10 \
            --gpus 0 \
            --trunc-layer "$layer"_weights \
            --trunc-value $val 2>&1 | tee models/finetune/vgg-$layer-$val.log
    done
done
