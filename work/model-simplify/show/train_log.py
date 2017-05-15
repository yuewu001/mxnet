#! /usr/bin/env python
#################################################################################
#     File Name           :     show_log.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-24 21:44]
#     Last Modified       :     [2017-04-29 22:53]
#     Description         :     show training log
#################################################################################


import re
import sys
import os.path as osp
import fig

if len(sys.argv) != 2:
    print 'show_train.py log_file'
    sys.exit()

with open(sys.argv[1], 'r') as fh:
    content = fh.read()

pattern = re.compile('Epoch\[(\d+)\] Validation-accuracy=(\d+\.\d+)')
res =[(int(c[0]), float(c[1])) for c in pattern.findall(content)]

print res
print 'best result:', max(res, key=lambda x:x[1])

#sort by epoch
epochs,accuracies = zip(*sorted(res, key=lambda x: x[0]))

fig.plot([epochs], [accuracies], 'epochs', 'accuracy',
         output_path=osp.splitext(sys.argv[1])[0] + '.pdf',
        marker_size=2)
