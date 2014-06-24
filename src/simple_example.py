#!/usr/bin/env python
from networkx import *
from vbmod import *
import igraph

# Simple network that can be cut into 2 fairly of obvious groups
SIMPLE = '../dat/simple_example/simple.edgelist'

def igraph_read_edgelist(filename):
    # simple igraph analog to newtorkx's read_edgelist
    G = igraph.Graph()
    with open(SIMPLE) as f:
        edges = [tuple(x.strip().split(" ")) for x in f]
    vertices = list(set([i for s in edges for i in s]))
    G.add_vertices(vertices) 
    G.add_edges(edges)
    # optional to label the nodes in order
    G.vs["label"] = range(len(vertices))
    return G

# read in list of edges
G_nx = read_edgelist(SIMPLE)
G_ig = igraph_read_edgelist(SIMPLE)

# convert networkx graph object to sparse matrix
A_nx = to_scipy_sparse_matrix(G_nx)
A_ig = G_ig.get_adjacency()

# same adjacency representation
#print A_nx.todense()
#print A_ig

####################
# set up model
N=A_nx.shape[0]         # number of nodes
Kvec=range(2,10+1)      # range of K values over which to search

# hyperparameters for priors
net0={}
net0['ap0']=N*1.
net0['bp0']=1.
net0['am0']=1.
net0['bm0']=N*1.

# options
opts={}
opts['NUM_RESTARTS']=1450

# run vb
(net,net_K)=learn_restart(A_nx.tocsr(),Kvec,net0,opts)

# display figures
#restart_figs(A,net,net_K)
#show()

# extract results
#Q = net_K[5]['Q']
Q = net['Q']
Q.tolist()
Q = [x.tolist()[0] for x in Q]
ma = [x.index(max(x)) for x in Q]

# See the results
#for i,j  in zip(Q,ma):
#    print i, j

#########

deng = G_ig.community_fastgreedy()
clust_wiggy = igraph.VertexClustering(G_ig, ma)
#clust = deng.as_clustering()
#igraph.plot(clust)
igraph.plot(clust_wiggy)
