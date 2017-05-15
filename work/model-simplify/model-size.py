#! /usr/bin/env python
#################################################################################
#     File Name           :     model-size.py
#     Created By          :     yuewu
#     Creation Date       :     [2017-04-30 19:04]
#     Last Modified       :     [2017-04-30 20:41]
#     Description         :      
#################################################################################


from common import modelzoo, find_mxnet
import mxnet as mx
import sys
import numpy as np

if len(sys.argv) != 3:
    print 'Usage: model-size.py prefix epoch'
    sys.exit()

prefix = sys.argv[1]
epoch = int(sys.argv[2])
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

param_num = 0
print 'arg params'
param_sizes={}
for k,v in arg_params.iteritems():
    num = np.prod(v.shape)
    param_num += num
    param_sizes[k] = num

param_sizes=sorted(param_sizes.iteritems(), key=lambda x:x[0])
for k,v in param_sizes:
    print k, v


#print 'aux params'
#for k,v in aux_params.iteritems():
#    num = np.prod(v.shape)
#    param_num += num
#    print k, ': ', num
#
print 'total number: ', param_num
