#!/usr/bin/env python

import pickle
import cPickle
import numpy
import pystan
import sivel
import matplotlib.pyplot as plt

f = open('fix3_x1.pkl','rb')
(fit,_) = pickle.load(f)

N = 6
D = fit['Delta'].shape[1]

mega = []
for index in xrange(1,D):
   mega.append([fit['Delta'][:,index]-fit['Delta'][:,0], \
   fit['EW'][:,index,0] - fit['EW'][:,0,0], \
   fit['EW'][:,index,1] - fit['EW'][:,0,1], \
   fit['sivel'][:,index]- fit['sivel'][:,0], \
   fit['x1'][:,index] - fit['x1'][:,0], \
   fit['ev_sig']*fit['ev'][:,2]* (fit['mag_int_raw'][:,index]-fit['mag_int_raw'][:,0])])

D = D-1
mega = numpy.array(mega)
mega_orig = mega
mega = numpy.reshape(mega,(D*N,20000),order='F')
print "Showing that it is feature outside, SN inside"
print fit['Delta'][0,1]-fit['Delta'][0,0],fit['Delta'][0,2]-fit['Delta'][0,0], fit['EW'][0,1,0]-fit['EW'][0,0,0]
print mega[0,0],mega[1,0],mega[D,0]

meas = numpy.mean(mega,axis=1)
meascov = numpy.cov(mega,rowvar=True)

pkl_file = open('gege_data.pkl', 'r')
data = pickle.load(pkl_file)
pkl_file.close()
sivel,sivel_err,x1,x1_err,zcmb,zerr,_ = sivel.sivel(data)
use = numpy.isfinite(sivel)
zcmb=zcmb[use]
zcmb0=zcmb[0]
zcmb=zcmb[1:]

data = {'D': D, 'N': N, 'meas': meas, 'meascov': meascov, 'zcmb':zcmb, 'zcmb0':zcmb0  }

init = [{'snparameters' : numpy.mean(mega_orig[:,1:,:],axis=2), \
   'pv_sig' : 300./3e5, \
   'pv_unit': numpy.zeros(D),\
   'pv0_unit': 0,
   'alpha': numpy.zeros(5)
   } \
for _ in range(4)]


sm = pystan.StanModel(file='c2.stan')
control = {'stepsize':1}
fit = sm.sampling(data=data, iter=400, chains=4,control=control,init=init, thin=1)


output = open('c2.pkl','wb')
pickle.dump((fit.extract(),fit.get_sampler_params()), output, protocol=2)
output.close()
print fit