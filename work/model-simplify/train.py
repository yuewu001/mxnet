#! /usr/bin/env python
#################################################################################
#     File Name           :     train.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-12-21 13:57]
#     Last Modified       :     [2017-04-24 20:41]
#     Description         :      
#################################################################################

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

def download_cifar10():
    data_dir="data"
    fnames = (os.path.join(data_dir, "cifar10_train.rec"),
              os.path.join(data_dir, "cifar10_val.rec"))
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec', fnames[1])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fnames[0])
    return fnames

if __name__ == '__main__':
    # download data
    (train_fname, val_fname) = download_cifar10()

    # parse args
    parser = argparse.ArgumentParser(description="train cifar10",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 1)

    parser.add_argument('--trunc-layer', type=str, nargs='+',
                       help='name of layers to truncate')
    parser.add_argument('--trunc-value', type=float,
                       help='truncate threshold')
    parser.set_defaults(
        # network
        network        = 'vgg_cifar',
        #num_layers     = 110,
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        mean_img       = 'data/mean.bin',
        num_classes    = 10,
        num_examples   = 50000,
        image_shape    = '3,32,32',
        pad_size       = 0,
        random_crop    = 0,
        # train
        batch_size     = 128,
        num_epochs     = 200,
        lr             = 1e-1,
        lr_factor      = 1e-1,
        lr_step_epochs = '50,100,150',
        wd             = 0.0005,
        #wd             = 0,
        mom            = 0.9,
        #mom            = 0,
        optimizer      = 'sgd',
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbol.'+args.network)
    sym = net.get_symbol(args.num_classes)

    # train
    fit.fit(args, sym, data.get_rec_iter)
