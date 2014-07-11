#!/usr/bin/env python

from networkx import *
from vbmod import *

# read in list of edges
G=read_edgelist('../data/football/football.edgelist')

# convert networkx graph object to sparse matrix
A=to_scipy_sparse_matrix(G)
N=A.shape[0]        # number of nodes

# hyperparameters for priors
net0={}
net0['ap0']=N*1.
net0['bp0']=1.
net0['am0']=1.
net0['bm0']=N*1.

# options
opts={}
opts['NUM_RESTARTS']=25
vbmod = mpl_vbmod.MPL_Vbmod(10,16,net0,opts)
# run vb
(net,net_K)=vbmod.learn_restart(A.tocsr())
# display figures
vbmod.restart_figs(A,net,net_K)
vbmod.show()

