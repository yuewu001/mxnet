#! /bin/bash
#################################################################################
#     File Name           :     run2.sh
#     Created By          :     yuewu
#     Creation Date       :     [2017-02-14 11:37]
#     Last Modified       :     [2017-02-15 08:42]
#     Description         :
#################################################################################

for layer in nw4_1 nw4_2 nw4_3 nw5_1 nw5_2 nw5_3
do
    #for val in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    for val in 0.05 0.95
    do
        python simplify.py \
            --pretrained models/pretrain/cifar10 --load-epoch 175 \
            --model-prefix models/finetune/cifar10 \
            --gpus 1 \
            --trunc-layer "$layer"_weights \
            --trunc-value $val 2>&1 | tee models/finetune/pet-$layer-$val.log
    done
done
