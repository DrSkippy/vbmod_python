#!/usr/bin/env python
"""
Originally:
Copyright (C) 2007, 2008 Jake Hofman <jhofman@gmail.com>
Distributed under GPL 3.0
http://www.gnu.org/licenses/gpl.txt

Modifications:
Scott Hendrickson scott@drskippy.net 2014-06
"""
# import modules
import sys
from scipy import special
from scipy.misc import comb
from scipy.special import digamma, betaln, gammaln
from scipy.sparse import lil_matrix
from numpy import *
from scipy import weave
import struct
import resource
#import subprocess
import numexpr
import logging

# set up logging
LOGFILENAME = "../vbmod-log"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s'
        , level=logging.INFO
        , filename=LOGFILENAME)
logr = logging.getLogger("vbmod_logger")

class Vbmod(object):

    net0_default = {
              "ap0":2
            , "bp0":1
            , "am0":1
            , "bm0":2
            , "a0" : None
            , "Q0" : None
                }   

    opts_default = {
            "TOL_DF":1e-2
            , "MAX_FITER":30
            , "VERBOSE":0
            , "SAVE_ITER":0
            , "NUM_RESTARTS":50
                }

    def __init__(self, k_min=2, k_max=6, net0={}, opts={}):
        """ 
        net0 and opts inputs are optional. if provided, length(net0.a0)
        must equal K.
        
        inputs:
         k_min: smallest k to search
         k_max: largest k to search
          net0: initialization/hyperparameter structure for network
                net0['Q0']: N-by-K initial mean-field matrix (rows sum to 1)
                net0['ap0']: alpha_{+0}, hyperparameter for prior on \theta_+
                net0['bp0']: beta_{+0}, hyperparameter for prior on \theta_+
                net0['am0']: alpha_{-0}, hyperparameter for prior on \theta_-
                net0['bm0']: beta_{-0}, hyperparameter for prior on \theta_-
                net0['a0']: alpha_{\mu0}, 1-by-K vector of hyperparameters for
                     prior on \pi
          opts: options
                opts['TOL_DF']: tolerance on change in F (outer loop)
                opts['MAX_FITER']: maximum number of F iterations (outer loop)
                opts['VERBOSE']: verbosity (0=quiet (default),1=print, 2=figures)
        """
        self.Kvec = array(range(k_min, k_max))
        logr.debug("kvec = {}".format(self.Kvec))
        logr.info("get options from opts struct opts= {}".format(opts))
        # make attributes for either defaults or opts passed to constructor
        for k in self.opts_default:
            if k in opts:
                setattr(self, k, opts[k])
            else:
                setattr(self, k, self.opts_default[k])
        logr.info("geting initial Q0 matrix and prior hyperparameters from net0= {}".format(net0))
        for k in self.net0_default:
            if k in net0:
                setattr(self, k, net0[k])
            else:
                setattr(self, k, self.net0_default[k])

    def init(self,N,K):
        """
        returns randomly-initialized mean-field matrix Q0 for vbmod. 
        
        inputs:
          N: number of nodes
          K: (maximum) number of modules
        
        outputs:
          Q: N-by-K mean-field matrix (rows sum to 1)
        """
        Q=mat(random.random([N,K]))
        Q=Q/(Q.sum(1)*ones([1,K]))
        return Q

    def rnd(self,N,K,tp,tm):
        """
        sample from vbmod likelihood. generates a random adjacency matrix
        sampled from a constrained stochastic block model specified by the
        given parameters.
        
        inputs:
          N: number of nodes
          K: number of modules
          tp: \theta_+, probability of edge within modules; tp=prob(Aij=1|zi=zj)
          tm: \theta_-, probability of edge between modules; tm=prob(Aij=1|zi!=zj)
        
        outputs:
          A: N-by-N adjacency matrix (logical, sparse)
        """
        # jntj: need pivec in here too
        mask=matrix(kron(eye(K),ones([N/K,N/K])))
        Qtrue=matrix(kron(eye(K),ones([N/K,1])))

        # jntj: very slow ... 
        A=multiply(tp>random.random([N,N]),mask)+multiply(tm>random.random([N,N]),1-mask)
        A=triu(A,1)
        A=A+A.transpose()
        
        A=lil_matrix(A,dtype='b')
        A=A.tocsr()
        return A

    def learn(self,A,K):
        """
        runs variational bayes for inference of network modules
        (i.e. community detection) under a constrained stochastic block
        model.
        
        net0 and opts inputs are optional. if provided, length(net0.a0)
        must equal K.
        
        inputs:
          A: N-by-N undirected (symmetric), binary adjacency matrix w/o
             self-edges (note: fastest for sparse and logical A)
          K: (maximum) number of modules
        
        outputs:
          net: posterior structure for network
            net['F']: converged free energy (same as net.F_iter(end))
            net['F_iter']: free energy over iterations (learning curve)
            net['Q']: N-by-K mean-field matrix (rows sum to 1)
            net['K']: K, passed for compatibility with vbmod_restart
            net['ap']: alpha_+, hyperparameter for posterior on \theta_+
            net['bp']: beta_+, hyperparameter for posterior on \theta_+
            net['am']: alpha_-, hyperparameter for posterior on \theta_-
            net['bm']: beta_-, hyperparameter for posterior on \theta_-
            net['a']: alpha_{\mu}, 1-by-K vector of hyperparameters for
                     posterior on \pi
        """
        N=A.shape[0]		# number of nodes
        M=0.5*A.sum(0).sum(1)  # total number of non-self edges
        M=M[0,0]
        C=comb(N,2)	   # total number of possible edges between N nodes
        
        uk=mat(ones([K,1]))
        un=mat(ones([N,1]))
        
        if self.a0 is None:
            a0_ = ones([1,K])
        else:
            a0_ = self.a0
        #
        if self.Q0 is None:
            Q = self.init(N, K)
        else:
            Q = self.Q0
        Qmat=mat(Q)
        logr.info("size of Q={}".format(Q.shape))
        logr.info("intialize variational distribution hyperparameters to be equal to prior hyperparameters")
        ap=self.ap0
        bp=self.bp0
        am=self.am0
        bm=self.bm0
        a=a0_
        n=Q.sum(0)
        # ensure a0 is a 1-by-K vector
        assert(a.shape == (1,K))

        # vector to store free energy over iterations
        F=[]
        for i in range(self.MAX_FITER):
            ####################
            #VBE-step, to update mean-field Q matrix over module assignments
            ####################
            
            # compute local and global coupling constants, JL and JG and
            # chemical potentials -lnpi
            psiap=digamma(ap)   # c+
            psibp=digamma(bp)   # d+
            psiam=digamma(am)   # c-
            psibm=digamma(bm)   # d-
            psip=digamma(ap+bp) # c+ and c-
            psim=digamma(am+bm) # d+ and d-
            JL=psiap-psibp-psiam+psibm  # c+ - d+ - c- + d-
            JG=psibm-psim-psibp+psip    # d- - (d+ and d-) - d+ + (c+ and c-) 

            lnpi=digamma(a)-digamma(sum(a))

            # local update (technically correct, but slow)
            for l in range(N):
                # exclude Q[l,:] from contributing to its own update
                Q[l,:]=zeros([1,K])
                # jntj: doesn't take advantage of sparsity
                Al=mat(A.getrow(l).toarray())
                AQl=multiply((Al.T*uk.T),Q).sum(0)
                nl=Q.sum(0)
                lnQl=JL*AQl-JG*nl+lnpi
                lnQl=lnQl-lnQl.max()
                Q[l,:]=exp(lnQl)
                Q[l,:]=Q[l,:]/Q[l,:].sum()

            ####################
            #VBM-step, update distribution over parameters
            ####################
            
            # compute expected occupation numbers <n*>s
            n=Qmat.sum(0)
        
            npp=0.5*(Qmat.T*A*Qmat).diagonal().sum()
            npm=0.5*trace(Qmat.T*(un*n-Qmat))-npp
            nmp=M-npp
            nmm=C-M-npm	 
        
            # compute hyperparameters for beta and dirichlet distributions over
            # theta_+, theta_-, and pi_mu
            ap=npp+self.ap0
            bp=npm+self.bp0
            am=nmp+self.am0
            bm=nmm+self.bm0
            a=n+a0_
            logr.info("ap={} bp={} am={} bm={} a={}".format( ap, bp, am, bm, a))

            # evaluate variational free energy, an approximation to the
            # negative log-evidence	
            ## Below 1e-323, numpy.log() return -inf which makes the rest of the
            ## method to somehow fail.
            Qmat[Qmat<1e-323] = 1e-323
            F.append(
                betaln(ap,bp)
                -betaln(self.ap0,self.bp0)
                +betaln(am,bm)
                -betaln(self.am0,self.bm0)
                +sum(gammaln(a))
                    -gammaln(sum(a))
                    -(sum(gammaln(a0_))
                        -gammaln(sum(a0_)))
                    -sum(multiply(Qmat,log(Qmat))))
            F[i]=-F[i]
            logr.info("iteration: {}, F={}".format(i+1, F[i]))
            # F should always decrease
            if (i>1) and F[i] > F[i-1]:
                print >>sys.stderr,"\twarning: F increased from", F[i-1] ,"to", F[i]
            if (i>1) and (abs(F[i]-F[i-1]) < self.TOL_DF):
                break
        return dict(F=F[-1],F_iter=F,Q=Q,K=K,ap=ap,bp=bp,am=am,bm=bm,a=a)

    def learn_restart(self,A):
        """
        runs vbmod with multiple restarts over a range of K values. (see
        vbmod for further documentation.) returns the best run over each
        K value (i.e. corresponding to the lowest variational free
        energy) as well as the best run over all K values.
        
        net0 and opts inputs are optional. F_K and net_K outputs are also
        optional.
        
        inputs: see learn() above
        
        outputs:
          net: posterior structure for best run over all K and
              restarts. see vbmod for further documentation.
          net_K: length-K array of posterior structures for best run over
              each K
        """
        N=A.shape[0]
        len_Kvec=len(self.Kvec)
        print >>sys.stderr,"running vbmod for ", self.NUM_RESTARTS ,"restarts"
        print >>sys.stderr, "N =", N , "K =", self.Kvec
        logr.info("running vbmod for {} restarts;  N = {}; K = {}".format(self.NUM_RESTARTS, N,  self.Kvec))
        net_K = []
        F_K = zeros(len_Kvec)
        
        for kndx in range(len_Kvec):		
            K=self.Kvec[kndx];
            print >>sys.stderr, "K=",K
            logr.info("K={}".format(K))
            net_KR = []
            F_KR=zeros(self.NUM_RESTARTS)
            for r in range(self.NUM_RESTARTS):
                #net_KR.append(learn(A,K,net0,opts))
                #net_KR.append(self.learn(A,K,net0,opts))
                net_KR.append(self.learn(A,K))
                F_KR[r]=net_KR[r]['F']

            print >>sys.stderr, "find best run for this value of K"
            (rndx,)=where(F_KR==F_KR.min())
            rndx=rndx[0]
            net_K.append(net_KR[rndx])
            F_K[kndx]=net_K[kndx]['F']
            print >>sys.stderr, "best run for K = {}; F = {}".format(K, F_K[kndx])
            logr.info("best run for K = {}; F = {}".format(K, F_K[kndx]))
        # find best run over all K values
        (kndx,)=where(F_K==F_K.min())
        kndx=kndx[0]
        net=net_K[kndx]
        net['K']=self.Kvec[kndx]
        logr.info("minimum at K = {} of F = {}".format(net['K'], net['F']))
        print >>sys.stderr, "minimum at K = {} of F = {}".format(net['K'], net['F'])
        return net, net_K

