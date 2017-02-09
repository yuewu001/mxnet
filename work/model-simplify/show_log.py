#! /usr/bin/env python
#################################################################################
#     File Name           :     show_log.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-24 21:44]
#     Last Modified       :     [2017-01-11 14:34]
#     Description         :
#################################################################################


import re
import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print  'Usage: show_log log [range]'
    sys.exit()

start_range = 0
if len(sys.argv) == 3:
    start_range = int(sys.argv[2])


path = sys.argv[1]

pattern = re.compile('Epoch\[(\d+)\] Validation-accuracy=(\d+\.\d+)')

with open(sys.argv[1], 'r') as fh:
    content = fh.read()

#accuracy
res = []
for c in pattern.findall(content):
    res.append((int(c[0]), float(c[1])))
res = sorted(res, key=lambda x: x[0])
epochs = [item[0] for item in res]
accuracies = [item[1] for item in res]

print 'Accuracy: ', max(res, key=lambda x: x[1])
plt.plot(epochs[start_range:], accuracies[start_range:])

#sparsity
pattern = re.compile('Epoch\[(\d+)\] Sparsity of (.+)_weights: (\d+\.\d+)')
res = []
for c in pattern.findall(content):
    res.append((int(c[0]), c[1], float(c[2])))
res = sorted(res, key=lambda x: x[0])

sparsity_dict = {}
for item in res:
    if item[1] not in sparsity_dict:
        sparsity_dict[item[1]] = {'epochs':[], 'sparsities':[]}

    sparsity_dict[item[1]]['epochs'].append(item[0])
    sparsity_dict[item[1]]['sparsities'].append(item[2])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
lines = []
legends = []
for k,v in sparsity_dict.iteritems():
    line, = ax.plot(v['epochs'][start_range:], v['sparsities'][start_range:])
    lines.append(line)
    legends.append(k)

ax.legend(lines,legends)

plt.show()
