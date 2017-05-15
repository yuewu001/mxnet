#! /usr/bin/env python
#################################################################################
#     File Name           :     analyze_layer.py
#     Created By          :     lixiaodan
#     Creation Date       :     [2017-04-30 10:57]
#     Last Modified       :     [2017-04-30 12:01]
#     Description         :     analyze per-layer accuracy
#################################################################################


import re
import sys
import numpy as np
import os.path as osp

if len(sys.argv) != 2:
    print 'analyze_layer.py algo'
    sys.exit()

sparsities = [0.05] + np.linspace(0.1,0.9,9).tolist() + [0.95]
layers = ['nw1_1', 'nw1_2', 'nw2_1', 'nw2_2', 'nw3_1', 'nw3_2', 'nw3_3', 'nw4_1', 'nw4_2', 'nw4_3', 'nw5_1', 'nw5_2', 'nw5_3']

algorithms=['trunc','pet','sofs','sofsrand']
assert(sys.argv[1] in algorithms)

if sys.argv[1] == 'trunc':
    layers=[v.replace('nw','conv') for v in layers]

pattern = re.compile('Epoch\[(\d+)\] Validation-accuracy=(\d+\.\d+)')
xs = []
ys = []
res = {}
for layer in layers:
    res[layer] = []
    for sparsity in reversed(sparsities):
        logfile='models/finetune/%s-%s-%g.log' %(sys.argv[1],layer,sparsity)
        assert(osp.exists(logfile) == True)

        with open(logfile, 'r') as fh:
            content = fh.read()

        #accuracy
        matches = []
        for c in pattern.findall(content):
            matches.append((int(c[0]), float(c[1])))
        matches = sorted(matches, key=lambda x: x[0])
        epochs = [item[0] for item in matches]
        accuracies = [item[1] for item in matches]
        assert(len(accuracies) != 0)
        res[layer].append(accuracies[-1])

thresh = 0.9278 * 0.99
print thresh
for layer in layers:
    #print res[layer]
    for k in xrange(len(sparsities)-1,-1,-1):
        if res[layer][k] > thresh:
            print 'layer %s, sparsity %g, accu %f' %(layer, sparsities[k], res[layer][k])
            break
