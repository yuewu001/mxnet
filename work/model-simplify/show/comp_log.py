#! /usr/bin/env python
#################################################################################
#     File Name           :     show_log.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-24 21:44]
#     Last Modified       :     [2017-04-30 11:26]
#     Description         :
#################################################################################


import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import fig

if len(sys.argv) != 3:
    print 'comp_log.py algo1 algo2'
    sys.exit()

def load_log(algo, layer, sparsities):
    pattern = re.compile('Epoch\[(\d+)\] Validation-accuracy=(\d+\.\d+)')

    res = []
    for sparsity in reversed(sparsities):
        path = 'models/finetune/' + algo + '-' + layer + '-' + str(sparsity) + '.log'
        assert(osp.exists(path))

        with open(path, 'r') as fh:
            content = fh.read()

        #accuracy
        matches = []
        for c in pattern.findall(content):
            matches.append((int(c[0]), float(c[1])))
        matches = sorted(matches, key=lambda x: x[0])
        accuracies = [item[1] for item in matches]
        assert(len(accuracies) != 0)

        res.append(accuracies[-1])

    return res


algos = [sys.argv[1], sys.argv[2]]
layers = ['nw1_1', 'nw1_2', 'nw2_1', 'nw2_2', 'nw3_1', 'nw3_2', 'nw3_3', 'nw4_1', 'nw4_2', 'nw4_3', 'nw5_1', 'nw5_2', 'nw5_3']
sparsities = [0.05] + np.linspace(0.1,0.9,9).tolist() + [0.95]

for layer in layers:
    xs=[sparsities for i in xrange(2)]
    ys=[]

    for algo in algos:
        if algo == 'trunc':
            ys.append(load_log(algo, layer.replace('nw','conv'), sparsities))
        else:
            ys.append(load_log(algo, layer, sparsities))

    fig.plot(xs, ys, 'sparsity', 'accuracy', legends=algos, clip_on=True, title=layer)
