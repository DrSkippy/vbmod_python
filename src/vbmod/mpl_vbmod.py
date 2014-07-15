#!/usr/bin/env python
"""
Originally:
Copyright (C) 2007, 2008 Jake Hofman <jhofman@gmail.com>
Distributed under GPL 3.0
http://www.gnu.org/licenses/gpl.txt

Modifications:
Scott Hendrickson scott@drskippy.net 2014-06

This extention of the Vbmod object enables plotting of intermediate steps
"""
# import modules
from vbmod import *

from scipy import special
from scipy.misc import comb
from scipy.special import digamma, betaln, gammaln
from scipy.sparse import lil_matrix
from numpy import *
from scipy import weave
import struct
import resource
import subprocess
import numexpr

logr = logging.getLogger("vbmod_logger")

from pylab import spy, show, imshow, axis, plot, figure, subplot, xlabel, ylabel, title, grid, hold, legend
from matplotlib.ticker import FormatStrFormatter

class MPL_Vbmod(Vbmod):

    def restart_figs(self,A,net,net_K):
        """
        plots results from vbmod_restart
        
        inputs:
          A: N-by-N undirected (symmetric), binary adjacency matrix w/o
              self-edges (note: fastest for sparse and logical A)
          net: posterior structure for best run over all K and
              restarts. see vbmod for further documentation.
          net_K: length-K array of posterior structures for best run over
              each K; see vbmod_restart for further documentation
        """
        
        N=net['Q'].shape[0]
        K=net['Q'].shape[1]
        figure()
        
        subplot(1,3,1)
        Kvec=[]
        F_K=[]
        for n in net_K:
            Kvec.append(n['K'])
            F_K.append(n['F'])
        Kvec=array(Kvec)
        F_K=array(F_K)

        plot(Kvec,F_K,'b^-')
        hold(True)

        plot([K],[net['F']],'ro',label='Kvb')
        hold(False)
        legend()
        title('complexity control')
        xlabel('K')
        ylabel('F')
        grid('on')
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        subplot(1,3,2)
        imshow(array(net['Q']),interpolation='nearest',aspect=(1.0*K)/N)
        title('Qvb')
        xlabel('K')
        ylabel('N')

        subplot(1,3,3)
        plot(arange(1,len(net['F_iter'])+1),net['F_iter'],'bo-')
        title('learning curve')
        xlabel('iteration')
        ylabel('F')
        grid('on')

    def show(self):
        show()
