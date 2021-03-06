#!/usr/bin/env python

import pickle
import pystan
import numpy
import scipy
import matplotlib.pyplot as plt

# input from data
pkl_file = open('gege_data.pkl', 'r')
data = pickle.load(pkl_file)
pkl_file.close()

# input from color analysis
pkl_file = open('fix3_x1.pkl', 'r')
(color,_)  = pickle.load(pkl_file)
pkl_file.close()

# input from results
f = open('c2.pkl','rb')
fit = pickle.load(f)

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

dm = color['Delta']-color['Delta'][:,0][:,None]
dm = dm[:,1:]
dm = dm[:,inda]
dm_meas = numpy.mean(dm,axis=0)
dm_cov = numpy.cov(dm,rowvar=False)


mn = fit['mn_Delta'][:,inda]
mn_meas = numpy.mean(mn,axis=0)
mn_cov = numpy.cov(mn,rowvar=False)

res = dm_meas-mn_meas
res_cov = dm_cov + mn_cov

def view():
    pkl_file = open('step.pkl', 'r')
    (data,_) = pickle.load(pkl_file)
    pkl_file.close()

    (low,lowm,lowp)= numpy.percentile(data['steplow'],(50,50-34,50+34))
    (high,highm, highp) = numpy.percentile(data['steplow']+data['stepdelta'],(50,50-34,50+34))

    mn =numpy.mean(data['stepdelta'])
    std= numpy.std(data['stepdelta'])
    print r"${:7.3f} \pm {:7.3f}$".format(mn,std)
    (mn,mnm,mnp) = numpy.percentile(data['stepdelta'],(50,50-34,50+34))
    print mn, mn-mnm, mnp-mn
    ax = plt.errorbar(mass, res, marker='o',linestyle="None",yerr = [numpy.sqrt(numpy.diag(res_cov)), numpy.sqrt(numpy.diag(res_cov))])
    plt.hlines(low, 7,10)
    plt.hlines(lowm, 7,10,color='red')
    plt.hlines(lowp, 7,10,color='red')
    plt.hlines(high, 10,13)
    plt.hlines(highm, 10,13,color='red')
    plt.hlines(highp, 10,13,color='red')
    plt.errorbar(12.5,high,yerr=[[mn-mnm, mnp-mn]],ecolor='black',elinewidth=3)
    plt.ylabel(r'Relative magnitude offset $\vec{\Delta}_{.0} - \mu[0:N]$ (mag)')
    plt.xlabel(r'$\log{(M_{\mathrm{host}}/M_{\odot})}$')
    plt.savefig("mass.pdf",bbox_inches='tight')

    from chainconsumer import ChainConsumer
    c = ChainConsumer()
    c.add_chain([data['steplow'],data['stepdelta'],data['mass_mn'],2*numpy.tan(data['mass_unif'])], parameters= \
        [r"$m_{low}$", r"$\Delta_{\mathrm{high}}$", r"$\langle M_{\mathrm{host}} \rangle$", r"$\sigma_{M_{\mathrm{host}}}$"],name='Master')
    fig =  c.plotter.plot(figsize="page")
    fig.savefig("mass_fit.pdf",bbox_inches='tight')

def fit():
    data = {'D' : len(res),\
        'res': res,\
        'rescov': res_cov,\
        'mass': mass,\
        'emass': emass}

    nchain =8
    init = [{
        'mass_0' : mass,\
        'steplow': 0,\
        'stephigh': 0,\
        'mass_mn': 10,\
        'mass_unif': 1 \
       } \
    for _ in range(nchain)]

    sm = pystan.StanModel(file='step.stan')
    control = {'stepsize':1}
    fit = sm.sampling(data=data, iter=2000, chains=nchain,control=control,init=init, thin=1)

    output = open('step.pkl','wb')
    pickle.dump((fit.extract(),fit.get_sampler_params()), output, protocol=2)
    output.close()
    print fit

view()
# fit()