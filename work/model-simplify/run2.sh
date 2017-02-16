#! /bin/bash
#################################################################################
#     File Name           :     run2.sh
#     Created By          :     yuewu
#     Creation Date       :     [2017-02-14 11:37]
#     Last Modified       :     [2017-02-16 11:39]
#     Description         :
#################################################################################

for layer in conv4_1 conv4_2 conv4_3 conv5_1 conv5_2 conv5_3
do
    for val in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    #for val in 0.05 0.95
    do
        python comp1.py \
            --pretrained models/pretrain/cifar10 --load-epoch 175 \
            --model-prefix models/finetune/cifar10 \
            --gpus 1 \
            --trunc-layer "$layer"_weight \
            --trunc-value $val 2>&1 | tee models/finetune/comp1-$layer-$val.log
    done
done
