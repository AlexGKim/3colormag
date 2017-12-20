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

cauchy_tau = numpy.array([54., 14, 40, 2, 0.058])

f = open('c2.pkl','rb')
(fit,_) = pickle.load(f)

mega = numpy.concatenate((fit['alpha'],0.08*(1+fit['dm_sig_unif'][:,None])),axis=1)

c = ChainConsumer()
c.add_chain(mega, parameters= \
    [r"$\alpha_{EW_{Ca}}$", r"$\alpha_{EW_{Si}}$", r"$\alpha_{\lambda_{Si}}$", r"$\alpha_{x_1}$", r"$\alpha_{A_{V,p}}$",r'$\sigma_M$'],name='Master')
fig =  c.plotter.plot(figsize="column", truth=numpy.zeros(6))
fig.savefig("example.pdf",bbox_inches='tight')

print numpy.std(fit['alpha'],axis=0)

table = c.analysis.get_latex_table(caption="Results for the tested models", label="tab:example")
print(table)


c= ChainConsumer()
c.add_chain(numpy.concatenate((fit['snp_mn'],cauchy_tau*numpy.tan(fit['L_snp_sig_unif'])),axis=1), parameters= \
    [r"$\langle {EW_{Ca}}\rangle$", r"$\langle{EW_{Si}}\rangle$", \
    r"$\langle{\lambda_{Si}}\rangle$", r"$\langle{x_1}\rangle$", r"$\langle{A_{V,p}}\rangle$",     r"$\sigma_{{EW_{Ca}}}$", r"$\sigma_{{EW_{Si}}}$", r"$\sigma_{{\lambda_{Si}}}$", r"$\sigma_{{x_1}}$", r"$\sigma_{{A_{V,p}}}$"], \
    name='Servant')
fig = c.plotter.plot(figsize="column", truth=numpy.zeros(10))
for ax in fig.axes:
    ax.xaxis.set_tick_params(labelsize=7)
    ax.xaxis.label.set_size(7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.yaxis.label.set_size(7)
fig.savefig("example2.pdf",bbox_inches='tight')

# c= ChainConsumer()
# c.add_chain(numpy.concatenate((fit['snp_sig_unif'],fit['L_snp_sig_unif']),axis=1), parameters= \
#     [r"$\langle {EW_{Ca}}\rangle$", r"$\langle{EW_{Si}}\rangle$", \
#     r"$\langle{\lambda_{Si}}\rangle$", r"$\langle{x_1}\rangle$", r"$\langle{A_{V,p}}\rangle$", \
#     r"$\sigma_{{EW_{Ca}}}$", r"$\sigma_{{EW_{Si}}}$", r"$\sigma_{{\lambda_{Si}}}$", r"$\sigma_{{x_1}}$", r"$\sigma_{{A_{V,p}}}$"], \
#     name='Servant')
# fig = c.plotter.plot(figsize="column", truth=numpy.zeros(10))
# for ax in fig.axes:
#     ax.xaxis.set_tick_params(labelsize=7)
#     ax.xaxis.label.set_size(7)
#     ax.yaxis.set_tick_params(labelsize=7)
#     ax.yaxis.label.set_size(7)
# fig.savefig("example3.pdf",bbox_inches='tight')

# c= ChainConsumer()
# c.add_chain(fit['snp_sig'], parameters= \
#     ["snpsig1", "snpsig2", "snpsig3", "snpsig4", "snpsig5"],name='Servant')
# c.plotter.plot(filename="example3.png", figsize="column", truth=numpy.zeros(5))


