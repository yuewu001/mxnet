#! /bin/bash
#################################################################################
#     File Name           :     simplify_trunc.sh
#     Created By          :     lixiaodan
#     Creation Date       :     [2017-04-30 12:11]
#     Last Modified       :     [2017-04-30 12:18]
#     Description         :      
#################################################################################

gpu=1
algo=trunc

#nw1_1 -> nw1_2
pretrained="models/finetune/$algo-conv1_1-0.7"

for percent in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
do
    modelfile=models/finetune-all/$algo-conv1_2-$percent-0240.params
    if [ ! -f $modelfile  ]; then
        python trunc.py \
            --pretrained $pretrained --load-epoch 220 \
            --model-prefix models/finetune-all/$algo-conv1_2-$percent \
            --gpus $gpu \
            --num-epochs 240 \
            --trunc-layer "conv1_2_weight" \
            --trunc-value $percent 2>&1 | tee models/finetune-all/"$algo"-conv1_2_weights-$percent.log
    else
        echo "$modelfile already exists"
    fi
done
