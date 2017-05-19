#! /bin/bash
#################################################################################
#     File Name           :     run.sh
#     Created By          :     yuewu
#     Creation Date       :     [2017-02-09 16:31]
#     Last Modified       :     [2017-05-16 11:45]
#     Description         :
#################################################################################

if [ $# -ne 2 ]; then
    echo "simplify-by-layer.sh head/tail pet/sofs/sofsrand/trunc"
    exit
fi

if [ "$1" = "head" ]; then
    layers="nw1_1 nw1_2 nw2_1 nw2_2 nw3_1 nw3_2 nw3_3"
    gpu=0
elif [ "$1" = "tail" ]; then
    layers="nw4_1 nw4_2 nw4_3 nw5_1 nw5_2 nw5_3"
    gpu=1
else
    echo "simplify-by-layer.sh head/tail pet/sofs/sofsrand/trunc"
    exit
fi

for layer in $layers
do
    for percent in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
    do
        if [ "$2" = "trunc" ]; then
            layer=$(echo $layer | sed -e "s/nw/conv/g")
            modelfile=models/finetune/"$2"/$layer-$percent-0220.params
            if [ ! -f $modelfile  ]; then
                mkdir -p models/finetune/"$2"/
                python trunc.py \
                    --pretrained models/pretrain/vgg-cifar10 --load-epoch 200 \
                    --model-prefix models/finetune/"$2"/$layer-$percent \
                    --gpus $gpu \
                    --save-period 220 \
                    --trunc-layer "$layer"_weight \
                    --trunc-value $percent 2>&1 | tee models/finetune/"$2"/$layer-$percent.log
            else
                echo "$modelfile already exists"
            fi
        else
            modelfile=models/finetune/"$2"/$layer-$percent-0220.params
            if [ ! -f $modelfile  ]; then
                mkdir -p models/finetune/"$2"/
                python simplify.py \
                    --pretrained models/pretrain/vgg-cifar10 --load-epoch 200 \
                    --model-prefix models/finetune/"$2"/$layer-$percent \
                    --optimizer "$2" \
                    --gpus $gpu \
                    --save-period 220 \
                    --trunc-layer "$layer"_weights \
                    --trunc-value $percent 2>&1 | tee models/finetune/"$2"/$layer-$percent.log
            else
                echo "$modelfile already exists"
            fi
        fi
    done
done
