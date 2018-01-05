#!/usr/bin/env python

import pickle
import cPickle
import numpy
import pystan
import sivel

# input from first try
f = open('c2.pkl','rb')
first = pickle.load(f)

# input from previous fit
f = open('fix3_x1.pkl','rb')
(inputfit,_) = pickle.load(f)

# scaling parameters
# cauchy_tau = numpy.array([54., 14, 40, 2, 0.058])
# alpha_scale = numpy.array([1.9e-3, 7.2e-3, 2.6e-3, 0.051, 1.7])  # 1/snparameters_sig*.05
param_sd = numpy.array([ 27,  6.9,  20,   0.98,   0.029])
alpha_scale = 0.05 /param_sd

# cauchy_tau = 4 * param_sd;              
# cauchy_tau[-1] = cauchy_tau[-1] / 75


# input from data
pkl_file = open('gege_data.pkl', 'r')
data = pickle.load(pkl_file)
pkl_file.close()
sivel,sivel_err,x1,x1_err,zcmb,zerr,_ = sivel.sivel(data)

# parameters
N = 6
D = inputfit['Delta'].shape[1]

# construct data vector from previous fit subtracting zeroth SN
mega = []
for index in xrange(1,D):
   mega.append([inputfit['Delta'][:,index]-inputfit['Delta'][:,0], \
   inputfit['EW'][:,index,0] - inputfit['EW'][:,0,0], \
   inputfit['EW'][:,index,1] - inputfit['EW'][:,0,1], \
   inputfit['sivel'][:,index]- inputfit['sivel'][:,0], \
   inputfit['x1'][:,index] - inputfit['x1'][:,0], \
   inputfit['ev_sig']*inputfit['ev'][:,2]* (inputfit['mag_int_raw'][:,index]-inputfit['mag_int_raw'][:,0])])
mega = numpy.array(mega)

D = D-1

# Initial condition is that Delta is partially due to intrinsic dispersion ...
dm = numpy.mean(inputfit['Delta'],axis=0)
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
print inputfit['Delta'][0,1]-inputfit['Delta'][0,0],inputfit['Delta'][0,2]-inputfit['Delta'][0,0], inputfit['EW'][0,1,0]-inputfit['EW'][0,0,0]
print mega[0,0],mega[1,0],mega[D,0]

meas = numpy.mean(mega,axis=1)
meascov = numpy.cov(mega,rowvar=True)

zcmb0=zcmb[0]
zcmb=zcmb[1:]

zerr0=zerr[0]
zerr = zerr[1:]

# get Childress Masses
f = open('MJC_compile_SNdata.pkl','r')
gal= pickle.load(f)

# match objects with masses
names= numpy.array(data['snlist'])

inda = []
mass = []
emass = []
for ind, nam in enumerate(names):
   if nam in gal.keys():
      inda.append(ind)
      mass.append(gal[nam]['Mass'])
      emass.append(gal[nam]['eMass'])

inda = numpy.array(inda)
emass = numpy.array(emass)
mass = numpy.array(mass)

# the first galaxy has a mass
print 'first in list with a host mass ', inda[0]
mass0 = mass[0]
emass0 = emass[0]

# what was the second sn indexed by 1 is no the first supernova indexed by 0 in python but by 1 in STAN
inda = inda[1:]
mass = mass[1:]
emass=emass[1:]
indapy = inda-1

def view():
   import matplotlib.pyplot as plt

   pkl_file = open('c2mag.pkl', 'r')
   data = pickle.load(pkl_file)
   pkl_file.close()

   mn=numpy.zeros((data['alpha'].shape[0], len(inda)))
   for i in xrange(len(inda)):
      for j in xrange(data['alpha'].shape[0]):
        mn[j,i] = numpy.dot(data['alpha'][j,:]*alpha_scale[:-1],data['snparameters'][j,indapy[i],:-1])
   mn_meas = numpy.mean(mn,axis=0)
   mn_cov = numpy.cov(mn,rowvar=False)

   dm = inputfit['Delta']-inputfit['Delta'][:,0][:,None]
   dm = dm[:,1:]
   dm = dm[:,indapy]
   dm_meas = numpy.mean(dm,axis=0)
   dm_cov = numpy.cov(dm,rowvar=False)

   res = dm_meas-mn_meas
   res_cov = dm_cov + mn_cov

   ax = plt.errorbar(mass, res, marker='o',linestyle="None",xerr=[emass,emass],yerr = [numpy.sqrt(numpy.diag(res_cov)), numpy.sqrt(numpy.diag(res_cov))])
   x=numpy.arange(6.8,13,0.05)
   (step,stepm,stepp)= numpy.percentile(data['stephigh'],(50,50-34,50+34),axis=0)/2
   zterm = numpy.mean(data['zeroterm'])
   plt.plot(x,step*numpy.tanh(10*(x-10))-zterm,color='black')
   plt.plot(x,stepm*numpy.tanh(10*(x-10))-zterm,linestyle=':',color='r')
   plt.plot(x,stepp*numpy.tanh(10*(x-10))-zterm,linestyle=':',color='g')
   plt.ylabel(r'Relative magnitude offset $\vec{\Delta}_{.0} - \mu[0:N]$ (mag)')
   plt.xlabel(r'$\log{(M_{\mathrm{host}}/M_{\odot})}$')
   plt.xlim((6.8,13))
   plt.savefig("mass.pdf",bbox_inches='tight')

   plt.clf()
   (fitmass,fitmassm,fitmassp) = numpy.percentile(data['mass_0'],(50,50-34,50+34),axis=0)
   plt.errorbar(mass,fitmass,xerr=[emass,emass],yerr=[fitmass-fitmassm, fitmassp-fitmass],linestyle='None',alpha=0.5,color='b')
   plt.plot(mass,fitmass,'o',linestyle='None',markersize=4,color='b')
   plt.plot([7.1,12.5],[7.1,12.5])
   plt.xlabel(r'$\log{(M_{\mathrm{host}}/M_{\odot})}$')
   plt.ylabel(r'$\theta_{M_{\mathrm{host}}}$')
   plt.savefig("masses.pdf",bbox_inches='tight')

   plt.clf()
   plt.plot(data['steplow'],label='steplow')
   plt.plot(data['stephigh'],label='stephigh')
   # plt.plot(data['stepnone'],label='stepnone')
   plt.legend()
   plt.savefig("masschain.pdf",bbox_inches='tight')

   from chainconsumer import ChainConsumer
   c = ChainConsumer()

   # c.add_chain([data['steplow'],data['stephigh'],stepdelta,data['stepnone'],data['mass_mn'],2*numpy.tan(data['mass_unif'])], parameters= \
   #    [r"$m_{\mathrm{low}}$", r"$m_{\mathrm{high}}$", r"$m_{\mathrm{high}}-m_{\mathrm{low}}$", r"$m_{\mathrm{none}}$", r"$\langle M_{\mathrm{host}} \rangle$", r"$\sigma_{M_{\mathrm{host}}}$"],name='Master')
   c.add_chain([data['steplow'],data['stephigh']], parameters= \
      [r"$m_{\mathrm{low}}$", r"$m_{\mathrm{high}}$"],name='Master')
   fig =  c.plotter.plot(figsize="page")
   fig.savefig("mass_fit.pdf",bbox_inches='tight')

def run():    
   data = {'D': D, 'N': N, 'meas': meas, 'meascov': meascov, 'zcmb':zcmb, 'zcmb0':zcmb0, \
      'zerr':zerr, 'zerr0': zerr0, 'param_sd': param_sd, 'D_gal': len(mass), 'inda': inda, 'mass': mass, 'emass': emass, 'mass0': mass0, 'emass0': emass0}

   firstL_snp_sig_unif= first['L_snp_sig_unif']



   nchain =4

   #initial condition for mass trimming extreme cases for intitial condition
   tanh_mass_0_init = [numpy.tanh(6*(numpy.random.normal(mass,emass)-10)) for _ in range(nchain)]
   tanh_mass_0_init = numpy.array(tanh_mass_0_init)
   tanh_mass_0_init[tanh_mass_0_init ==1] = 1-1e-16
   tanh_mass_0_init[tanh_mass_0_init ==-1] = -1+1e-16

   init = [{
      # 'alpha': numpy.zeros(N-1) , \
      # 'pv_unit': pv_prior[1:] ,\
      # 'pv0_unit': pv_prior[0] ,\
      # 'dm_sig_unif' : 0., \
      # 'dm_unit':   dm_prior[1:],\
      # 'dm0_unit':  dm_prior[0] ,\
      # # # 'z_true': zcmb, \
      # # # 'z0_true': zcmb0, \
      # 'snparameters_alpha' :snparameters_alpha_prior,\
      'L_snp_cor': numpy.identity(N-1), \
      # # first sigmas are cauchy_tau*0.25
      # 'L_snp_sig_unif': numpy.zeros(N-1)+numpy.arctan(0.25) ,\
      # 'snp_mn': snparameters_mn \
      # 'snparameters': snparameters\
      'alpha': first['alpha'].mean(axis=0)+numpy.random.normal(0,first['alpha'].std(axis=0)), \
      'pv_unit': first['pv_unit'].mean(axis=0)+numpy.random.normal(0,first['pv_unit'].std(axis=0)), \
      'pv0_unit': first['pv0_unit'].mean(axis=0)+numpy.random.normal(0,first['pv0_unit'].std(axis=0)),\
      'dm_sig_unif' : first['dm_sig_unif'].mean(axis=0), \
      'dm_unit':   first['dm_unit'].mean(axis=0)+numpy.random.normal(0,first['dm_unit'].std(axis=0)),\
      'dm0_unit':  first['dm0_unit'].mean(axis=0)+numpy.random.normal(0,first['dm0_unit'].std(axis=0)),\
      'snparameters_alpha' :first['snparameters_alpha'].mean(axis=0)+numpy.random.normal(0,first['snparameters_alpha'].std(axis=0)),\
      'L_snp_sig_unif': firstL_snp_sig_unif.mean(axis=0),\
      'snp_mn': first['snp_mn'].mean(axis=0)+numpy.random.normal(0,first['snp_mn'].std(axis=0)), \
      'mass_0' : mass,\
      'mass0_0' : mass0,\
      'steplow': 0,\
      'stephigh': 0.02\
      # 'mass_mn': 10,\
      # 'mass_unif': 0.5 \
      } \
   for _ in range(nchain)]

   sm = pystan.StanModel(file='c2mag.stan')
   control = {'stepsize':1}
   fit = sm.sampling(data=data, iter=500, chains=nchain,control=control,init=init, thin=1)

   output = open('c2mag.pkl','wb')
   pickle.dump(fit.extract(), output, protocol=2)
   output.close()
   print fit

view()
# run()
# 