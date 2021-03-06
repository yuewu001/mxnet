#! /usr/bin/env python
#################################################################################
#     File Name           :     train.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-12-21 13:57]
#     Last Modified       :     [2017-05-17 09:26]
#     Description         :
#################################################################################

import os
import sys
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

    parser.add_argument('--pretrained', type=str, required=True,
                        help='the pre-trained model')
    parser.add_argument('--trunc-layer', type=str, nargs='+',
                       help='name of layers to truncate')
    parser.add_argument('--trunc-value', type=float, nargs='+',
                       help='truncate threshold')

    parser.set_defaults(
        # network
        network        = 'vgg_cifar_s2',
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
        num_epochs     = 220,
        lr             = 1e-2,
        lr_factor      = 2,
        wd             = 0,
        mom            = 0
    )
    args = parser.parse_args()
    algorithms = ['pet', 'sofs', 'sofsrand']
    if args.optimizer not in algorithms:
        logging.error("optimizer %s not allowd, choices = %s", args.optimizer, str(algorithms))
        sys.exit()

    if args.optimizer == 'pet':
        args.network = 'vgg_cifar_s'

    # load pretrained model
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.pretrained, args.load_epoch)

    # load network
    from importlib import import_module
    net = import_module('symbol.'+args.network)
    new_sym = net.get_symbol(args.num_classes)

    # train
    args.truncates = {}
    assert(len(args.trunc_layer) == len(args.trunc_value))
    for trunc_layer,trunc_value in zip(args.trunc_layer, args.trunc_value):
        args.truncates[trunc_layer] = trunc_value
    #args.trunc_threshs = {'nw1_1_weights':0.1}
    #args.trunc_percent = {'nw1_1_weights':0.1}
    #args.trunc_threshs = 1e-3 #1e-5
    fit.fit(args        = args,
            network     = new_sym,
            data_loader = data.get_rec_iter,
            arg_params  = arg_params,
            aux_params  = aux_params)
