#! /usr/bin/env python
#################################################################################
#     File Name           :     test.py
#     Created By          :     lixiaodan
#     Creation Date       :     [2017-04-30 11:28]
#     Last Modified       :     [2017-04-30 11:56]
#     Description         :
#################################################################################

import argparse
import logging
import score
import mxnet as mx
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--data-val', type=str, default='data/cifar10_val.rec')
    parser.add_argument('--mean-img', type=str, default='data/mean.bin')
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-shape', type=str, default='3,32,32')
    parser.add_argument('--data-nthreads', type=int, default=4,
                        help='number of threads for data decoding')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    metrics = [mx.metric.create('acc')]

    sparsities = [0.05] + np.linspace(0.1,0.9,9).tolist() + [0.95]
    layers = ['nw1_1', 'nw1_2', 'nw2_1', 'nw2_2', 'nw3_1', 'nw3_2', 'nw3_3', 'nw4_1', 'nw4_2', 'nw4_3', 'nw5_1', 'nw5_2', 'nw5_3']
    algorithms=['trunc','pet','sofs','sofsrand']

    assert(args.algo in algorithms)
    res={}
    if args.algo == 'trunc':
        layers=[l.replace('nw', 'conv') for l in layers]
    for layer in layers:
        res[layer] = []
        for sparsity in sparsities:
            prefix='models/finetune/%s-%s-%g' %(args.algo,layer,sparsity)
            print prefix

            (speed,) = score.score(model_prefix=prefix, epoch=args.epoch,
                             data_val=args.data_val, mean_img=args.mean_img,
                             metrics=metrics, gpus=args.gpus,
                             batch_size=args.batch_size,
                             image_shape=args.image_shape,
                             data_nthreads=args.data_nthreads)
            logging.info('Finished with %f images per second', speed)

            for m in metrics:
                logging.info(m.get())
                res[layer].append(m.get()[1])

    output_file='models/finetune/%s_result.txt' %(args.algo)
    logging.info('save result to %s', output_file)
    with open(output_file, 'w') as fh:
        for layer in layers:
            val_accu = [str(v) for v in reversed(res[layer])]
            fh.write('%s\t%s\n' %(layer, '\t'.join(val_accu)))
