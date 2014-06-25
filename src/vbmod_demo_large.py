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

def demo_largeN(N=1e3,Kvec=array([4,3,5]),ktot=16,kout=6):
	"""
	function to demonstrate vbmod for larger number of nodes
	"""
	K=Kvec[0]
	Kvec=sort(Kvec)
	#pivec=ones(1,Ktrue)/Ktrue;

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
	tp=(ktot-kout)/(float(N)/K-1)
	tm=kout/(float(N)*(K-1)/K)

	print "generating random adjacency matrix ... "
	# slow right now
	A=rnd(N,K,tp,tm)

	print "running variational bayes ... "
	t=time()
	(net,net_K)=learn_restart(A,Kvec,net0,opts)
	print "finished in", time()-t , "seconds"
	print "displaying results ... "
	restart_figs(A,net,net_K)
	show()

if __name__ == '__main__':
	demo_largeN()
