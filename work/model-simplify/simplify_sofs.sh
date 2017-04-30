#! /bin/bash
#################################################################################
#     File Name           :     run.sh
#     Created By          :     yuewu
#     Creation Date       :     [2017-02-09 16:31]
#     Last Modified       :     [2017-04-30 12:18]
#     Description         :
#################################################################################

gpu=0
algo=sofs

#nw1_1 -> nw1_2
pretrained="models/finetune/$algo-nw1_1-0.6"

for percent in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
do
    modelfile=models/finetune-all/$algo-nw1_2-$percent-0240.params
    if [ ! -f $modelfile  ]; then
        python simplify.py \
            --pretrained $pretrained --load-epoch 220 \
            --model-prefix models/finetune-all/$algo-nw1_2-$percent \
            --optimizer sofs \
            --gpus $gpu \
            --num-epochs 240 \
            --trunc-layer "nw1_2_weights" \
            --trunc-value $percent 2>&1 | tee models/finetune-all/"$algo"-nw1_2_weights-$percent.log
    else
        echo "$modelfile already exists"
    fi
done
