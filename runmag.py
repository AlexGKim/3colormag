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
(fit,_) = pickle.load(f)

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

def view():
   pkl_file = open('c2mag.pkl', 'r')
   (data,_) = pickle.load(pkl_file)
   pkl_file.close()

   mn=numpy.zeros((data['alpha'].shape[0], len(inda)))
   for i in xrange(len(inda)):
      for j in xrange(fit['alpha'].shape[0]):
        mn[j,i] = numpy.dot(fit['alpha'][j,:]*alpha_scale[:-1],fit['snparameters'][j,inda[i],:-1])
   mn_meas = numpy.mean(mn,axis=0)
   mn_cov = numpy.cov(mn,rowvar=False)

   dm = fit['Delta']-color['fit'][:,0][:,None]
   dm = dm[:,1:]
   dm = dm[:,inda]
   dm_meas = numpy.mean(dm,axis=0)
   dm_cov = numpy.cov(dm,rowvar=False)

   res = dm_meas-mn_meas
   res_cov = dm_cov + mn_cov

   (low,lowm,lowp)= numpy.percentile(data['steplow'],(50,50-34,50+34))
   (high,highm, highp) = numpy.percentile(data['stephigh'],(50,50-34,50+34))

   stepdelta =  data['stephigh'] - data['steplow']
   mn =numpy.mean(stepdelta)
   std= numpy.std(stepdelta)
   print r"${:7.3f} \pm {:7.3f}$".format(mn,std)
   (mn,mnm,mnp) = numpy.percentile(stepdelta,(50,50-34,50+34))
   print mn, mn-mnm, mnp-mn
   ax = plt.errorbar(mass, res, marker='o',linestyle="None",xerr=[emass,emass],yerr = [numpy.sqrt(numpy.diag(res_cov)), numpy.sqrt(numpy.diag(res_cov))])
   plt.hlines(low, 6.8,10)
   plt.hlines(lowm, 6.8,10,color='red')
   # plt.hlines(lowp, 6.8,10,color='red')
   plt.hlines(high, 10,12.4)
   plt.hlines(highm, 10,12.4,color='red')
   plt.hlines(highp, 10,12.4,color='red')
   plt.errorbar(12.75,high,yerr=[[mn-mnm, mnp-mn]],ecolor='black',elinewidth=3,marker='o')
   plt.ylabel(r'Relative magnitude offset $\vec{\Delta}_{.0} - \mu[0:N]$ (mag)')
   plt.xlabel(r'$\log{(M_{\mathrm{host}}/M_{\odot})}$')
   plt.xlim((6.8,13))
   plt.savefig("mass.pdf",bbox_inches='tight')

   plt.clf()
   (fitmass,fitmassm,fitmassp) = numpy.percentile(data['mass_0'],(50,50-34,50+34),axis=0)
   plt.errorbar(mass,fitmass,xerr=[emass,emass],yerr=[fitmass-fitmassm, fitmassp-fitmass],linestyle='None')
   plt.plot([7.1,12.5],[7.1,12.5])
   plt.xlabel(r'$\log{(M_{\mathrm{host}}/M_{\odot})}$')
   plt.ylabel(r'$\theta_{M_{\mathrm{host}}}$')
   plt.savefig("masses.pdf",bbox_inches='tight')

   from chainconsumer import ChainConsumer
   c = ChainConsumer()
   c.add_chain([data['steplow'],data['stephigh'],data['stepnone'],stepdelta,data['mass_mn'],2*numpy.tan(data['mass_unif'])], parameters= \
      [r"$m_{\mathrm{low}}$", r"$m_{\mathrm{high}}$" r"$m_{\mathrm{high}}-m_{\mathrm{low}}$", r"$m_{\mathrm{none}}$", r"$\langle M_{\mathrm{host}} \rangle$", r"$\sigma_{M_{\mathrm{host}}}$"],name='Master')
   fig =  c.plotter.plot(figsize="page")
   fig.savefig("mass_fit.pdf",bbox_inches='tight')

def run():    
   data = {'D': D, 'N': N, 'meas': meas, 'meascov': meascov, 'zcmb':zcmb, 'zcmb0':zcmb0, \
      'zerr':zerr, 'zerr0': zerr0, 'param_sd': param_sd, 'D_gal': len(mass), 'inda': inda+1, 'mass': mass, 'emass': emass}


   firstL_snp_sig_unif= first['L_snp_sig_unif']
   firstL_snp_sig_unif[:,-1] = numpy.pi/8

   nchain =4
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
      'dm_sig_unif' : first['dm_sig_unif'].mean(axis=0)+numpy.random.normal(0,first['dm_sig_unif'].std(axis=0)), \
      'dm_unit':   first['dm_unit'].mean(axis=0)+numpy.random.normal(0,first['dm_unit'].std(axis=0)),\
      'dm0_unit':  first['dm0_unit'].mean(axis=0)+numpy.random.normal(0,first['dm0_unit'].std(axis=0)),\
      'snparameters_alpha' :first['snparameters_alpha'].mean(axis=0)+numpy.random.normal(0,first['snparameters_alpha'].std(axis=0)),\
      'L_snp_sig_unif': firstL_snp_sig_unif.mean(axis=0)+numpy.random.normal(0,firstL_snp_sig_unif.std(axis=0)),\
      'snp_mn': first['snp_mn'].mean(axis=0)+numpy.random.normal(0,first['snp_mn'].std(axis=0)), \
      'mass_0' : mass,\
      'steplow': 0,\
      'stepnone': 0,\
      'stephigh': 0,\
      'mass_mn': 10,\
      'mass_unif': 1 \
      } \
   for _ in range(nchain)]

   sm = pystan.StanModel(file='c2mag.stan')
   control = {'stepsize':1}
   fit = sm.sampling(data=data, iter=1000, chains=nchain,control=control,init=init, thin=1)

   output = open('c2mag.pkl','wb')
   pickle.dump(fit.extract(), output, protocol=2)
   output.close()
   print fit

run()
