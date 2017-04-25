#! /usr/bin/env python
#################################################################################
#     File Name           :     show_log.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-24 21:44]
#     Last Modified       :     [2017-04-24 16:43]
#     Description         :
#################################################################################


import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import fig


pattern = re.compile('Epoch\[(\d+)\] Validation-accuracy=(\d+\.\d+)')

sparsities = [0.05] + np.linspace(0.1,0.9,9).tolist() + [0.95]

layers = ['nw1_1', 'nw1_2', 'nw2_1', 'nw2_2', 'nw3_1', 'nw3_2', 'nw3_3', 'nw4_1', 'nw4_2', 'nw4_3', 'nw5_1', 'nw5_2', 'nw5_3']
#layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']

#for k,v in sparsity_dict.iteritems():
#    line, = ax.plot(v['epochs'][start_range:], v['sparsities'][start_range:])
#    lines.append(line)
#    legends.append(k)

prefixes = [sys.argv[1], sys.argv[2]]

for layer in layers:
    xs = []
    ys = []
    for prefix in prefixes:
        x_values = []
        y_values = []
        for sparsity in sparsities:
            path = prefix + '-' + layer + '-' + str(sparsity) + '.log'

            if osp.exists(path) == False:
                continue

            with open(path, 'r') as fh:
                content = fh.read()

            #accuracy
            res = []
            for c in pattern.findall(content):
                res.append((int(c[0]), float(c[1])))
            res = sorted(res, key=lambda x: x[0])
            epochs = [item[0] for item in res]
            accuracies = [item[1] for item in res]
            if len(accuracies) == 0:
                continue
            x_values.append(1-sparsity)
            y_values.append(accuracies[-1])

        xs.append(x_values)
        ys.append(y_values)

    fig.plot(xs, ys, 'sparsity', 'accuracy', legends=prefixes,
             output_path="sofs-cnn-simplify.pdf",
             ylim=[0.86,0.93],clip_on=True, title=layer)
