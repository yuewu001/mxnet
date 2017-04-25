import argparse
from common import modelzoo, find_mxnet
import mxnet as mx
import time
import os
import logging


def score(model_prefix, epoch, data_val, mean_img, metrics, gpus, batch_size,
          image_shape='3,32,32', data_nthreads=4):
    # create data iterator
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    data = mx.io.ImageRecordIter(
        path_imgrec        = data_val,
        label_width        = 1,
        mean_img           = mean_img,
        preprocess_threads = data_nthreads,
        batch_size         = batch_size,
        data_shape         = data_shape,
        rand_crop          = False,
        rand_mirror        = False)

    # create module
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]

    mod = mx.mod.Module(symbol=sym, context=devs)
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)
    if not isinstance(metrics, list):
        metrics = [metrics,]
    tic = time.time()
    num = 0
    for batch in data:
        mod.forward(batch, is_train=False)
        for m in metrics:
            mod.update_metric(m, batch.label)
        num += batch_size
    return (num / (time.time() - tic), )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--data-val', type=str, required=True)
    parser.add_argument('--model-prefix', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--mean-img', type=str, required=True)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-shape', type=str, default='3,32,32')
    parser.add_argument('--data-nthreads', type=int, default=4,
                        help='number of threads for data decoding')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k = 5)]

    (speed,) = score(metrics = metrics, **vars(args))
    logging.info('Finished with %f images per second', speed)

    for m in metrics:
        logging.info(m.get())
