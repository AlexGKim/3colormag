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

f = open('c2.pkl','rb')
(fit,_) = pickle.load(f)

mega = numpy.concatenate((fit['alpha'],fit['dm_sig'][:,None]),axis=1)

c = ChainConsumer()
c.add_chain(mega, parameters= \
    [r"$\alpha_{EW_{Ca}}$", r"$\alpha_{EW_{Si}}$", r"$\alpha_{\lambda_{Si}}$", r"$\alpha_{x_1}$", r"$\alpha_{A_{V,p}}$",r'$\sigma_M$'],name='Master')
c.plotter.plot(filename="example.png", figsize="column", truth=numpy.zeros(6))

table = c.analysis.get_latex_table(caption="Results for the tested models", label="tab:example")
print(table)