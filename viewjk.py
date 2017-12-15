#!/usr/bin/env python

import pickle
import pystan
import matplotlib.pyplot as plt
from matplotlib import rc
import corner
from matplotlib.backends.backend_pdf import PdfPages
import numpy
import sncosmo
import scipy
import cPickle
import matplotlib as mpl
import sivel

from chainconsumer import ChainConsumer

def one(index):
    f = open('jkout/jk{:03d}.pkl'.format(index),'rb')
    (fit,_) = pickle.load(f)

    print fit['residual'].mean(), fit['residual'].std()
    plt.hist(fit['residual'])
    plt.savefig(filename='jk.png')


    mega = numpy.concatenate((fit['alpha'],fit['dm_sig'][:,None]),axis=1)

    c = ChainConsumer()
    c.add_chain(mega, parameters= \
        [r"$\alpha_{EW_{Ca}}$", r"$\alpha_{EW_{Si}}$", r"$\alpha_{\lambda_{Si}}$", r"$\alpha_{x_1}$", r"$\alpha_{p}$",r'$\sigma_M$'])
    c.plotter.plot(filename="jkexample.png", figsize="column", truth=numpy.zeros(6))

def many(done):
    mn=[]
    sig = []
    sig_m = []
    for i in done:
        f = open('jkout/jk{:03d}.pkl'.format(i),'rb')
        (fit,_) = pickle.load(f)

        mn.append(fit['residual'].mean())

        # total noise
        sig.append(fit['residual'].std())

        # measured noise (without pv and dm)
        temp  = fit['delta_holdout']
        for n in xrange(5):
            temp = temp -  fit['alpha'][:,n] * fit['snparameters'][:,i-1,n] ; 
        sig_m.append(temp.std())

    mn=numpy.array(mn)
    sig=numpy.array(sig)
    sig_m=numpy.array(sig_m)

    f, axarr = plt.subplots(2, sharex=True)

    axarr[0].errorbar(xrange(len(done)),mn,yerr=[sig,sig],fmt='o',label='Total Uncertainty')
    axarr[0].errorbar(xrange(len(done)),mn,yerr=[sig_m,sig_m],linestyle='None',label='Measurement Uncertainty')
    axarr[0].set_xlim(-0.1, len(done)-.9)
    axarr[0].legend()
    axarr[0].set_ylabel(r'$\Delta_{.0}$ residual relative to SN 0 (mag)')

    m_w = numpy.sum(mn/sig**2) / numpy.sum(1/sig**2)
    pull = (mn-m_w)/sig
    axarr[1].plot(pull,marker='o',linestyle='None')
    axarr[1].axhline(1,linestyle='dashed')
    axarr[1].axhline(-1,linestyle='dashed')
    axarr[1].set_ylabel('pull')
    axarr[1].set_xlabel('SN index')
    plt.savefi('jkresiduals.png')

# one(1)
# wefwe

done = numpy.arange(168,171,dtype='int')
done = numpy.append(done,1)
many(done)
wefe


