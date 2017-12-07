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
mpl.rcParams['font.size'] = 28

f = open('c2.pkl','rb')
(fit,_) = pickle.load(f)

from chainconsumer import ChainConsumer
c = ChainConsumer()
c.add_chain(fit['alpha'], parameters=["$\alpha_1$", "$\alpha_2$", "$\alpha_3$", "$\alpha_4$", "$\alpha_5$"])
c.plotter.plot(filename="example.png", figsize="column", truth=mean)