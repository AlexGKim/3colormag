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

param_sd = numpy.array([ 27,  6.9,  20,   0.98,   0.029])

# input from data
pkl_file = open('gege_data.pkl', 'r')
data = pickle.load(pkl_file)
pkl_file.close()
sivel,sivel_err,x1,x1_err,zcmb,zerr,_ = sivel.sivel(data)

# input from color analysis
pkl_file = open('fix3_x1.pkl', 'r')
(color,_)  = pickle.load(pkl_file)
print type(color)
pkl_file.close()

# input from results
f = open('c2.pkl','rb')
(fit,_) = pickle.load(f)

def orig():
    f = open('c2.orig.pkl','rb')
    (fit,_) = pickle.load(f)
    c = ChainConsumer()
    c.add_chain(numpy.concatenate((fit['alpha'],fit['dm_sig'][:,None]),axis=1), parameters= \
        [r"$\alpha_{EW_{Ca}}$", r"$\alpha_{EW_{Si}}$", r"$\alpha_{\lambda_{Si}}$", r"$\alpha_{x_1}$", r"$\alpha_{A_{V,p}}$",r'$\sigma_M$'],name='Master')
    fig =  c.plotter.plot(figsize="column", truth=numpy.zeros(6))
    fig.savefig("top.orig.pdf",bbox_inches='tight')
    table = c.analysis.get_latex_table(caption="Results for the tested models", label="tab:example")
    print(table)
    dm = fit['dm_sig'][:,None]*fit['dm_unit']
    plt.hist([dm.flatten(),numpy.median(dm,axis=0)],bins=50,normed=True,label=['stack','median'])
    plt.legend()
    plt.xlabel(r'$dm$')
    plt.savefig("top_m.pdf",bbox_inches='tight')

    c= ChainConsumer()
    c.add_chain(numpy.median(fit['snparameters'],axis=0), \
        parameters= [r"$ {EW_{Ca}}$", r"${EW_{Si}}$", r"${\lambda_{Si}}$", r"${x_1}$", r"${A_{V,p}}$"], \
        name='Servant')
    fig = c.plotter.plot(figsize="column", truth=numpy.zeros(5))
    for ax in fig.axes:
        ax.xaxis.set_tick_params(labelsize=9)
        ax.xaxis.label.set_size(9)
        ax.yaxis.set_tick_params(labelsize=9)
        ax.yaxis.label.set_size(9)
    fig.savefig("top_snp.orig.pdf",bbox_inches='tight')
    table = c.analysis.get_latex_table(caption="Results for the tested models", label="tab:example")
    print(table)

def top():
    dm_sig = 0.1*(numpy.tan(fit['dm_sig_unif']))
    # dm_sig =fit['dm_sig_unif']
    c = ChainConsumer()
    # c.add_chain(numpy.concatenate((fit['alpha'],dm_sig[:,None]),axis=1), parameters= \
        # [r"$\alpha_{EW_{Ca}}$", r"$\alpha_{EW_{Si}}$", r"$\alpha_{\lambda_{Si}}$", r"$\alpha_{x_1}$", r"$\alpha_{A_{V,p}}$",r'$\sigma_M$'],name='Master')
    c.add_chain(numpy.concatenate((fit['alpha'],dm_sig[:,None]),axis=1), parameters= \
        [r"$\alpha_{EW_{Ca}}$", r"$\alpha_{EW_{Si}}$", r"$\alpha_{\lambda_{Si}}$", r"$\alpha_{x_1}$",r'$\sigma_M$'],name='Master')
    fig =  c.plotter.plot(figsize="column", truth=numpy.zeros(6))
    fig.savefig("top.pdf",bbox_inches='tight')
    table = c.analysis.get_latex_table(caption="Results for the tested models", label="tab:example")
    print(table)
    plt.clf()

    # plt.hist(((fit['z_true']-zcmb[None,1:])/zerr[None,1:]).flatten(),bins=50)
    # plt.xlabel(r'$z$ residual pull')
    # plt.ylabel(r'posterior stack')
    # plt.savefig("top_z.pdf",bbox_inches='tight')
    # plt.clf()

    # plt.plot(fit['dm_sig_unif'])

    dm = dm_sig[:,None]*fit['dm_unit']
    plt.hist([dm.flatten(),numpy.median(dm,axis=0)],bins=50,normed=True,label=['stack','median'])
    plt.legend()
    plt.xlabel(r'$dm$')
    plt.savefig("top_m.pdf",bbox_inches='tight')

    c= ChainConsumer()
    c.add_chain(numpy.median(fit['snparameters'],axis=0), \
        parameters= [r"$ {EW_{Ca}}$", r"${EW_{Si}}$", r"${\lambda_{Si}}$", r"${x_1}$", r"${A_{V,p}}$"], \
        name='Fit')
    table = c.analysis.get_latex_table(caption="Results for the tested models", label="tab:example")
    print(table)

    # # input from previous fit
    # f = open('fix3_x1.pkl','rb')
    # (ofit,_) = pickle.load(f)
    # D = ofit['Delta'].shape[1]

    # # construct data vector from previous fit subtracting zeroth SN
    # mega = []
    # for index in xrange(1,D):
    #    mega.append([ofit['Delta'][:,index]-ofit['Delta'][:,0], \
    #    ofit['EW'][:,index,0] - ofit['EW'][:,0,0], \
    #    ofit['EW'][:,index,1] - ofit['EW'][:,0,1], \
    #    ofit['sivel'][:,index]- ofit['sivel'][:,0], \
    #    ofit['x1'][:,index] - ofit['x1'][:,0], \
    #    ofit['ev_sig']*ofit['ev'][:,2]* (ofit['mag_int_raw'][:,index]-ofit['mag_int_raw'][:,0])])
    # mega = numpy.array(mega)

    # snparameters = numpy.mean(mega[:,1:,:],axis=2)  # shape (D,5)

    # c.add_chain(snparameters,name='Data')

    fig = c.plotter.plot(figsize="column", truth=numpy.zeros(5))
    for ax in fig.axes:
        ax.xaxis.set_tick_params(labelsize=9)
        ax.xaxis.label.set_size(9)
        ax.yaxis.set_tick_params(labelsize=9)
        ax.yaxis.label.set_size(9)
    fig.savefig("top_snp.pdf",bbox_inches='tight')
    plt.clf()


def population():
    c= ChainConsumer()
    c.add_chain(numpy.concatenate((fit['snp_mn'],fit['L_snp_sig_unif']),axis=1), parameters= \
        [r"$\langle {EW_{Ca}}\rangle$", r"$\langle{EW_{Si}}\rangle$", \
        r"$\langle{\lambda_{Si}}\rangle$", r"$\langle{x_1}\rangle$", r"$\langle{A_{V,p}}\rangle$", \
        r"$\sigma_{{EW_{Ca}}}$", r"$\sigma_{{EW_{Si}}}$", r"$\sigma_{{\lambda_{Si}}}$", r"$\sigma_{{x_1}}$", r"$\sigma_{{A_{V,p}}}$"], \
        name='Servant')
    table = c.analysis.get_latex_table(caption="Results for the tested models", label="tab:example")
    print(table)
    fig = c.plotter.plot(figsize="column", truth=numpy.zeros(10))
    for ax in fig.axes:
        ax.xaxis.set_tick_params(labelsize=7)
        ax.xaxis.label.set_size(7)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.yaxis.label.set_size(7)
    fig.savefig("population.pdf",bbox_inches='tight')

def childress():

    fiveoverlog10 = 5/numpy.log(10)

    dm = numpy.mean(color['Delta']-color['Delta'][:,0][:,None],axis=0)
    dm = dm[1:]
    mn = numpy.mean(fit['mn_Delta'],axis=0)

    res = dm-mn

    # get Childress Masses
    f = open('MJC_compile_SNdata.pkl','r')
    gal= pickle.load(f)

    # match objects with masses
    names= numpy.array(data['snlist'])[1:]

    i = numpy.intersect1d(names, gal.keys(), assume_unique=True)

    inda = numpy.zeros(len(i),dtype='int')    

    mass=[]
    emass = []
    for j in xrange(len(i)):
        inda[j] = numpy.where(names == i[j])[0]
        emass.append(gal[i[j]]['eMass'])
        mass.append(gal[i[j]]['Mass'])

    mass = numpy.array(mass)
    emass= numpy.array(emass)

    plt.scatter(mass, res[inda])
    plt.show()

# orig()
top()
population()
# childress()

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


