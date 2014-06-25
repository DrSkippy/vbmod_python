#!/usr/bin/env python
"""

Copyright (C) 2007, 2008 Jake Hofman <jhofman@gmail.com>
Distributed under GPL 3.0
http://www.gnu.org/licenses/gpl.txt

jntj: todo
  - pivec in rnd
  - make to a class, object oriented
  - switch for inline usage?
  - error checking
  - compatibility w/ scipy 0.7?

Modifictions Scott Hendrickson drskippy@twitter.com 2014-06

"""

# import modules
from vbmod import *

import networkx as nx
from time import *

def demo():
    """
    function to demonstrate vbmod
    """
    N=128.
    K=4
    #Kvec=array(range(K-2,K+2+1))
    ktot=16.
    kout=6.
    #pivec=ones(1,Ktrue)/Ktrue;     

    # determine within- and between- module edge probabilities from above
    tp=(ktot-kout)/(N/K-1)
    tm=kout/(N*(K-1)/K)
    
    vbmod = MPL_Vbmod(K-2, K+3)
    
    print "generating random adjacency matrix ... "
    A=vbmod.rnd(N,K,tp,tm)

    print "running variational bayes ... "
    t=time()
    #(net,net_K)=vbmod.learn_restart(A,Kvec)
    (net,net_K)=vbmod.learn_restart(A)
    print "finished in", time()-t , "seconds"
    print "displaying results ... "
    vbmod.restart_figs(A,net,net_K)
    show()
    return net

if __name__ == '__main__':
    demo()
