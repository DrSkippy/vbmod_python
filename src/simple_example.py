#!/usr/bin/env python
# use either networkx or igraph as desired
from vbmod import *

# Simple network that can be cut into 2 fairly of obvious groups
SIMPLE = '../data/simple_example/simple.edgelist'
KMIN, KMAX = 6,10

try:
    ##### IGRAPH #####
    import igraph
    from scipy import sparse

    # simple igraph analog to newtorkx's read_edgelist
    G_ig = igraph.Graph()
    with open(SIMPLE) as f:
        edges = [tuple(x.strip().split(" ")) for x in f]
    vertices = list(set([i for s in edges for i in s]))
    G_ig.add_vertices(vertices) 
    G_ig.add_edges(edges)
    # optional to label the nodes in order
    G_ig.vs["label"] = range(len(vertices))

    # convert igraph graph object to sparse matrix
    A_ig = G_ig.get_adjacency()
    A = sparse.csr_matrix(list(A_ig), A_ig.shape)
except ImportError:
    ##### NETWORKX #####
    from networkx import *

    # read in list of edges
    G_nx = read_edgelist(SIMPLE)
    # convert networkx graph object to sparse matrix
    A_nx = to_scipy_sparse_matrix(G_nx)
    A = A_nx.tocsr()


N=A.shape[0]         # number of nodes

# hyperparameters for priors
net0={}
net0['ap0']=N*1.
net0['bp0']=1.
net0['am0']=1.
net0['bm0']=N*1.

# options
opts={}
opts['NUM_RESTARTS']=50

# run vbmod
vbm = mpl_vbmod.MPL_Vbmod(KMIN, KMAX, net0, opts)
(net,net_K) = vbm.learn_restart(A)

# display figures
vbm.restart_figs(A,net,net_K)
vbm.show()

# extract results
Q = net['Q']
Q.tolist()
Q = [x.tolist()[0] for x in Q]
ma = [x.index(max(x)) for x in Q]

#################################
# This part requires igraph!
#################################
#deng = G_ig.community_fastgreedy()
#clust = deng.as_clustering()
clust_w = igraph.VertexClustering(G_ig, ma)
#igraph.plot(clust)
igraph.plot(clust_w)
