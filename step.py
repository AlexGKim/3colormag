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
(fit,_) = pickle.load(f)

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

print numpy.prod(numpy.diag(res_cov)), numpy.linalg.det(res_cov)/numpy.prod(numpy.diag(res_cov))
wefewf
# plt.errorbar(mass, res, marker='o',linestyle="None",yerr = [numpy.sqrt(numpy.diag(res_cov)), numpy.sqrt(numpy.diag(res_cov))])
# plt.show()
data = {'D' : len(res),\
    'res': res,\
    'rescov': res_cov,\
    'mass': mass,\
    'emass': emass}

nchain =4
init = [{
    'mass_0' : mass,\
    'steplow': 0,\
    'stephigh': 0,\
    'mass_mn': 10,\
    'mass_inif': 1 \
   } \
for _ in range(nchain)]

sm = pystan.StanModel(file='step.stan')
control = {'stepsize':1}
fit = sm.sampling(data=data, iter=2000, chains=nchain,control=control,init=init, thin=1)

output = open('step.pkl','wb')
pickle.dump((fit.extract(),fit.get_sampler_params()), output, protocol=2)
output.close()
print fit

