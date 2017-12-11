#!/usr/bin/env python

import pickle
import cPickle
import numpy
import pystan
import sivel
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

# input from previous fit
f = open('fix3_x1.pkl','rb')
(fit,_) = pickle.load(f)

# input from data
pkl_file = open('gege_data.pkl', 'r')
data = pickle.load(pkl_file)
pkl_file.close()
sivel,sivel_err,x1,x1_err,zcmb,zerr,_ = sivel.sivel(data)

# parameters
N = 6
D = fit['Delta'].shape[1]

# Initial condition is that Delta is entirely due to intrinsic dispersion
dm = numpy.mean(fit['Delta'],axis=0)

# Initial condition is that pv is partially responsible for the intrinsic dispersion
pv = numpy.sign(numpy.mean(fit['Delta'],axis=0))

# construct data vector from previous fit subtracting zeroth SN
mega = []
for index in xrange(1,D):
   mega.append([fit['Delta'][:,index]-fit['Delta'][:,0], \
   fit['EW'][:,index,0] - fit['EW'][:,0,0], \
   fit['EW'][:,index,1] - fit['EW'][:,0,1], \
   fit['sivel'][:,index]- fit['sivel'][:,0], \
   fit['x1'][:,index] - fit['x1'][:,0], \
   fit['ev_sig']*fit['ev'][:,2]* (fit['mag_int_raw'][:,index]-fit['mag_int_raw'][:,0])])
mega = numpy.array(mega)


D = D-1

# initial condition of the sn parameters are the measurement
snparameters = numpy.mean(mega[:,1:,:],axis=2)  # shape (D,5)

mega = numpy.reshape(mega,(D*N,mega.shape[2]),order='F')
print "Showing that it is (feature 1 SN1, feature 1 SN2, ... feature 1 SN D, feature 2 SN 1, ...)"
print fit['Delta'][0,1]-fit['Delta'][0,0],fit['Delta'][0,2]-fit['Delta'][0,0], fit['EW'][0,1,0]-fit['EW'][0,0,0]
print mega[0,0],mega[1,0],mega[D,0]

meas = numpy.mean(mega,axis=1)
meascov = numpy.cov(mega,rowvar=True)

zcmb0=zcmb[0]
zcmb=zcmb[1:]


data = {'D': D, 'N': N, 'meas': meas, 'meascov': meascov, 'zcmb':zcmb, 'zcmb0':zcmb0  }

nchain =8
init = [{'snparameters' : snparameters, \
   # 'pv_sig' : 300/3e5, \
   'pv_unit': pv[1:],\
   'pv0_unit': pv[0], \
   'dm_sig' : 0.08, \
   'dm_unit':  dm[1:]/0.08,\
   'dm0_unit': dm[0]/0.08, \
   'alpha': numpy.zeros(5)
   } \
for _ in range(nchain)]

sm = pystan.StanModel(file='c2.stan')
control = {'stepsize':1}
fit = sm.sampling(data=data, iter=4000, chains=nchain,control=control,init=init, thin=1)

output = open('c2.pkl','wb')
pickle.dump((fit.extract(),fit.get_sampler_params()), output, protocol=2)
output.close()
print fit