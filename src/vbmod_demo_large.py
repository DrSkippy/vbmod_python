#!/usr/bin/env python
"""
Copyright (C) 2007, 2008 Jake Hofman <jhofman@gmail.com>
Distributed under GPL 3.0
http://www.gnu.org/licenses/gpl.txt

Modifictions Scott Hendrickson drskippy@twitter.com 2014-06
"""
# import modules
from vbmod import *
import networkx as nx
from time import *

# Demonstrate vbmod for larger number of nodes
N=1e3
Ktarget=4
ktot=16
kout=6
# hyperparameters for priors
net0={}
net0['ap0']=N*2.
net0['bp0']=2.
net0['am0']=2.
net0['bm0']=N*2.
# options
opts={}
opts['NUM_RESTARTS']=1
opts['MAX_FITER']=50
# determine within- and between- module edge probabilities from above
tp=(ktot-kout)/(float(N)/Ktarget-1)
tm=kout/(float(N)*(Ktarget-1)/Ktarget)

vbm = mpl_vbmod.MPL_Vbmod(3,5, net0, opts)
print "generating random adjacency matrix ... "
# slow right now
A=vbm.rnd(N,Ktarget,tp,tm)
print "running variational bayes ... "
t=time()
(net,net_K)=vbm.learn_restart(A)
print "finished in", time()-t , "seconds"
print "displaying results ... "
vbm.restart_figs(A,net,net_K)
vbm.show()
