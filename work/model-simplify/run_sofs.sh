#! /bin/bash
#################################################################################
#     File Name           :     run.sh
#     Created By          :     yuewu
#     Creation Date       :     [2017-02-09 16:31]
#     Last Modified       :     [2017-02-19 15:41]
#     Description         :
#################################################################################

for layer in nw1_1 nw1_2 nw2_1 nw2_2 nw3_1 nw3_2 nw3_3
do
    #for val in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    for val in 0.05 0.95
    do
        python simplify2.py \
            --pretrained models/pretrain/cifar10 --load-epoch 175 \
            --model-prefix models/finetune/cifar10 \
            --gpus 0 \
            --trunc-layer "$layer"_weights \
            --trunc-value $val 2>&1 | tee models/finetune/sofs-$layer-$val.log
    done
done
