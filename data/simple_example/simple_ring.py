#!/usr/bin/env python
import itertools
groups = 8
gsize = 6
nodes = [i for i in range(groups*gsize)]
for g in range(groups):
    s = nodes[g*gsize:(g+1)*gsize]
    # connect the nodes in a group
    for i in itertools.combinations(s,2):
        print '{} {}'.format(*i)
    # connect the groups in a ring
    print '{} {}'.format(nodes[g*gsize], nodes[(g+1)*gsize % len(nodes)])
