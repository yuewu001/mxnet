#! /bin/bash
#################################################################################
#     File Name           :     run.sh
#     Created By          :     yuewu
#     Creation Date       :     [2017-02-09 16:31]
#     Last Modified       :     [2017-05-19 09:22]
#     Description         :
#################################################################################

if [ $# -ne 2 ]; then
    echo "simplify-all.sh pet/sofs/sofsrand/trunc layer"
    exit
fi

layer=$2
for percent in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
do
    if [ "$1" = "trunc" ]; then
        layer=$(echo $layer | sed -e "s/nw/conv/g")
        modelfile=models/finetune-all/"$1"/$layer-$percent-0220.params
        if [ ! -f $modelfile  ]; then
            mkdir -p models/finetune-all/"$1"/
            python trunc.py \
                --pretrained models/pretrain/vgg-cifar10 --load-epoch 200 \
                --model-prefix models/finetune-all/"$1"/$layer-$percent \
                --gpus 1 \
                --save-period 220 \
                --num-epochs 220 \
                --trunc-layer "conv1_1_weight" "conv3_1_weight" "conv3_3_weight" "conv4_1_weight" "conv4_2_weight" "conv4_3_weight" "conv5_1_weight" "$layer"_weight \
                --trunc-value 0.3 0.9 0.8 0.4 0.7 0.2 0.1 $percent 2>&1 | tee models/finetune-all/"$1"/$layer-$percent.log
        else
            echo "$modelfile already exists"
        fi
    else
        modelfile=models/finetune-all/"$1"/$layer-$percent-0220.params
        if [ ! -f $modelfile  ]; then
            mkdir -p models/finetune-all/"$1"/
            python simplify.py \
                --pretrained models/pretrain/vgg-cifar10 --load-epoch 200 \
                --model-prefix models/finetune-all/"$1"/$layer-$percent \
                --optimizer "$1" \
                --gpus 0 \
                --save-period 220 \
                --num-epochs 220 \
                --trunc-layer "nw1_1_weights" "nw1_2_weights"  "nw3_1_weights" "nw3_3_weights" "nw4_1_weights" "nw4_2_weights" "nw4_3_weights" "nw5_1_weights" "$layer"_weights \
                --trunc-value 0.4 0.9 0.95 0.9 0.6 0.4 0.5 0.3 $percent 2>&1 | tee models/finetune-all/"$1"/$layer-$percent.log
        else
            echo "$modelfile already exists"
        fi
    fi
done

#sofs records-0.915

#trunc records-0.915


#sofs records-0.92
#nw1_1 0.6
#nw1_2 0.1
#nw2_1 0
#nw2_2 0
#nw3_1 0.05
#nw3_2 0
#nw3_3 0.1
#nw4_1 0.4
#nw4_2 0.6
#nw4_3 0.5
#nw5_1 0.7
#nw5_2 0
#nw5_3 0.7


#trunc records-0.92
#conv1_2 0.7
#conv1_2 0
#conv2_1 0
#conv2_2 0
#conv3_1 0.1
#conv3_2 0
#conv3_3 0.2
#conv4_1 0.6
#conv4_2 0.3
#conv4_3 0.8
#conv5_1 0.9
#conv5_2 0
#conv5_3 0
