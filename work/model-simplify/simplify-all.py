#! /usr/bin/env python
#################################################################################
#     File Name           :     simplify-all.py
#     Created By          :     yuewu
#     Creation Date       :     [2017-05-19 09:23]
#     Last Modified       :     [2017-05-19 10:49]
#     Description         :
#################################################################################

import sys
import os
import os.path as osp
import numpy as np
import re

if len(sys.argv) < 2:
    print 'Usage: simplify-all.py pet/sofs/sofsrand/trunc [gpu=0]'
    sys.exit()

algorithms=['trunc','pet','sofs','sofsrand']
assert(sys.argv[1] in algorithms)

gpu = 0
if len(sys.argv) == 3:
    gpu=int(sys.argv[2])

sparsities = [0.05] + np.linspace(0.1,0.9,9).tolist() + [0.95]
layers = ['nw1_1', 'nw1_2', 'nw2_1', 'nw2_2', 'nw3_1', 'nw3_2', 'nw3_3', 'nw4_1', 'nw4_2', 'nw4_3', 'nw5_1', 'nw5_2', 'nw5_3']

if sys.argv[1] == 'trunc':
    cmd_prefix='python trunc.py'
    layers=[v.replace('nw','conv') for v in layers]
    layer_posix = '_weight'
else:
    cmd_prefix='python simplify.py --optimizer %s' %(sys.argv[1])
    layer_posix = '_weights'

records = {}
pretrained='models/pretrain/vgg-cifar10'
epoch=200
thresh = 0.915
for layer in layers:
    for sparsity in reversed(sparsities):
        print 'layer %s, sparsity %f: ' %(layer,sparsity)

        logdir='models/finetune-%f/%s/' %(thresh, sys.argv[1])
        if osp.exists(logdir) == False:
            os.makedirs(logdir)
        model_prefix=osp.join(logdir, '%s-%g' %(layer,1.0 - sparsity))
        logfile=model_prefix + '.log'

        cmd =  '{0} --pretrained \"{1}\" --load-epoch {2} --model-prefix \"{3}\" --gpus {4} --save-period {5} --num-epochs {5}'.format(cmd_prefix,
                                                                                                                                      pretrained,
                                                                                                                                      epoch,model_prefix,gpu,
                                                                                                                                      epoch+20)
        trunc_layers = []
        trunc_values = []
        for k,v in records.iteritems():
            trunc_layers.append(k + layer_posix)
            trunc_values.append(str(1.0 - v))
        trunc_layers.append(layer+layer_posix)
        trunc_values.append(str(1.0 - sparsity))

        cmd += ' --trunc-layer ' + ' '.join(trunc_layers)
        cmd += ' --trunc-value ' + ' '.join(trunc_values)

        cmd += ' 2>&1 | tee %s' %(logfile)
        if osp.exists(logfile) == False: 
            os.system(cmd)
        else:
            print '%s already exists' %(logfile)

        assert(osp.exists(logfile) == True)

        with open(logfile, 'r') as fh:
            content = fh.read()

        #accuracy
        matches = []
        pattern = re.compile('Epoch\[(\d+)\] Validation-accuracy=(\d+\.\d+)')
        for c in pattern.findall(content):
            matches.append((int(c[0]), float(c[1])))
        matches = sorted(matches, key=lambda x: x[0])
        epochs = [item[0] for item in matches]
        accuracies = [item[1] for item in matches]
        assert(len(accuracies) != 0)
        accuracy = accuracies[-1]

        print 'layer %s, sparsity %f, accuracy %f: ' %(layer,sparsity, accuracy)

        if accuracy >= thresh:
            records[layer] = sparsity
            print records
            record_file = osp.join(logdir, 'records.txt')
            with open(record_file, 'w') as fh:
                for k,v in records.iteritems():
                    fh.write('%s %f\n' %(k,v))
            break
    break
