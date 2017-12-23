#!/usr/bin/env python

import pickle
import cPickle
import numpy
import pystan
import sivel

# input from first try
# f = open('c2.pkl.orig','rb')
# (first,_) = pickle.load(f)

# input from previous fit
f = open('fix3_x1.pkl','rb')
(fit,_) = pickle.load(f)

# scaling parameters
# cauchy_tau = numpy.array([54., 14, 40, 2, 0.058])
# alpha_scale = numpy.array([1.9e-3, 7.2e-3, 2.6e-3, 0.051, 1.7])  # 1/snparameters_sig*.05
param_sd = numpy.array([ 27,  6.9,  20,   0.98,   0.029])
cauchy_tau = 4 * param_sd;              
cauchy_tau[-1] = cauchy_tau[-1] / 75


# input from data
pkl_file = open('gege_data.pkl', 'r')
data = pickle.load(pkl_file)
pkl_file.close()
sivel,sivel_err,x1,x1_err,zcmb,zerr,_ = sivel.sivel(data)

# parameters
N = 6
D = fit['Delta'].shape[1]

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

# Initial condition is that Delta is partially due to intrinsic dispersion ...
dm = numpy.mean(fit['Delta'],axis=0)
w = dm < 0.3
dm_prior = dm / dm[w].std()
w = dm > 0.3
dm_prior[w]=0

# Initial condition is that pv is partially responsible for the intrinsic dispersion
pv_prior = (zcmb*dm)/(zcmb*dm).std()
# ... except for extreme guys where it is entirely
pv_prior[w] = (numpy.log(10)/5*zcmb[w]*dm[w])/(300./3e5)

# initial condition of the sn parameters are the measurement
snparameters = numpy.mean(mega[:,1:,:],axis=2)  # shape (D,5)
snparameters_sig = numpy.std(snparameters,axis=0)
snparameters_mn = numpy.mean(snparameters,axis=0)

snparameters_alpha_prior = (snparameters - snparameters_mn[None,:])/snparameters_sig

mega = numpy.reshape(mega,(D*N,mega.shape[2]),order='F')
print "Showing that it is (feature 1 SN1, feature 1 SN2, ... feature 1 SN D, feature 2 SN 1, ...)"
print fit['Delta'][0,1]-fit['Delta'][0,0],fit['Delta'][0,2]-fit['Delta'][0,0], fit['EW'][0,1,0]-fit['EW'][0,0,0]
print mega[0,0],mega[1,0],mega[D,0]

meas = numpy.mean(mega,axis=1)
meascov = numpy.cov(mega,rowvar=True)

zcmb0=zcmb[0]
zcmb=zcmb[1:]

zerr0=zerr[0]
zerr = zerr[1:]
data = {'D': D, 'N': N, 'meas': meas, 'meascov': meascov, 'zcmb':zcmb, 'zcmb0':zcmb0, \
   'zerr':zerr, 'zerr0': zerr0, 'param_sd': param_sd}


nchain =4
init = [{
   'alpha': numpy.zeros(N-1) , \
   'pv_unit': pv_prior[1:] ,\
   'pv0_unit': pv_prior[0] ,\
   'dm_sig_unif' : 0., \
   'dm_unit':   dm_prior[1:],\
   'dm0_unit':  dm_prior[0] ,\
   # # 'z_true': zcmb, \
   # # 'z0_true': zcmb0, \
   'snparameters_alpha' :snparameters_alpha_prior,\
      #1./(N-1))*(snparameters - snparameters_mn[None,:])/(cauchy_tau*numpy.tan(0.2)), \
   # 'L_snp_cor': numpy.identity(N-1), \
   # first sigmas are cauchy_tau*0.25
   'L_snp_sig_unif': numpy.zeros(N-1)+numpy.arctan(0.25) ,\
   'snp_mn': snparameters_mn \
   # 'snparameters': snparameters\
   } \
for _ in range(nchain)]

sm = pystan.StanModel(file='c2.stan')
control = {'stepsize':1}
fit = sm.sampling(data=data, iter=2000, chains=nchain,control=control,init=init, thin=1)

output = open('c2.pkl','wb')
pickle.dump((fit.extract(),fit.get_sampler_params()), output, protocol=2)
output.close()
print fit
