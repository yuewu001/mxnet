#! /usr/bin/env python
#################################################################################
#     File Name           :     show/draw_net.py
#     Created By          :     lixiaodan
#     Creation Date       :     [2017-04-30 19:56]
#     Last Modified       :     [2017-04-30 20:04]
#     Description         :      
#################################################################################

from common import find_mxnet
import mxnet as mx
from importlib import import_module
from symbol import vgg_cifar

sym = vgg_cifar.get_symbol(10)

obj=mx.viz.plot_network(sym)
obj.render('vgg-bn')
