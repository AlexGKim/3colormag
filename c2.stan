data {
  int D;                                  // Number of supernova differences
  int N;                                  // Number of features + Delta

  vector[D*N] meas;                       // Feature Data
  matrix[D*N, D*N] meascov;               // Feature Data covariance
  vector[D] zcmb;                         // Redshift Data
  real zcmb0;
  vector[D] zerr;                         // Redshift Uncertainty UNUSED FOR THE MOMENT
  real zerr0;

  // vector[N-1] cauchy_tau;              // tau of the cauchy prior of distributions of parameters
  vector[N-1] param_sd;                   // standard deviation of parameters used set scales
}

transformed data{
  cholesky_factor_cov[D*N] L;             // Feature Covariance Cholesky factors
  real pv_sig;                            // hand-specified peculiar velocity stanadard deviation
  real fiveoverlog10;
  vector[N-1] cauchy_tau;                 // scale for parameter-distribution sigma
  vector[N-1] alpha_scale;                // scale for alpha
  real dm_mu;                            // scale of dm sigma cauchy
  real dm_tau;                            // scale of dm sigma cauchy

  L = cholesky_decompose(meascov);
  pv_sig=300./3e5;
  fiveoverlog10 = 5/log(10.);

  alpha_scale = 0.05 * inv(param_sd);
  alpha_scale[N-1] = alpha_scale[N-1] * 75;  // the p parameter standard deviation is noise not intrinsic dominated

  cauchy_tau = 4 * param_sd;              
  cauchy_tau[N-1] = cauchy_tau[N-1] /75; // the p parameter standard deviation is noise not intrinsic dominated

  dm_mu =0.08;
  dm_tau=0.08;
}

parameters {
  vector[N-1] alpha;                              // feature cofactors

  vector[D] pv_unit;                              // peculiar velocity pv_sig fixed 
  real pv0_unit;

  real<lower=-pi()/4, upper=pi()/2> dm_sig_unif;  // peculiar magnitude
  vector[D] dm_unit;
  real dm0_unit; 

  // vector[D] z_true;                               // CMB redshift UNUSED BECAUSE OF LARGE Z ERRORS
  // real z0_true;

  vector[N-1] snparameters_alpha[D];           // SN parameters in linear independent form
  // cholesky_factor_corr[N-1] L_snp_cor;         // correlation matrix of SN parameters
  vector<lower=0, upper=pi()/2>[N-1] L_snp_sig_unif;         // sigma of the covariance matrix for SN parameters
  vector[N-1] snp_mn;                          // distribution of parameters mn

  // vector[N-1] snparameters[D];
}

transformed parameters{
  vector[N-1] snparameters[D];      // SN parameters covariance term

  {
    // matrix[N-1, N-1] L_snp_cov;                   // covariance matrix of SN parameters
    // feature covariance
    // L_snp_cov = diag_pre_multiply(cauchy_tau .* tan(L_snp_sig_unif), L_snp_cor);
    for (d in 1:D){           //for each supernova pair
      // snparameters[d] =  snp_mn + L_snp_cov * snparameters_alpha[d];
      snparameters[d] =  snp_mn + cauchy_tau .* tan(L_snp_sig_unif) .* snparameters_alpha[d];
    }
  }
}

model {

  vector[D*N] mn;                               // model vector corresponding to data
  vector[D] pv;                                 // peculiar velocity
  real pv0;
  vector[D] pz;                                 // peculiar redshift
  real pz0;

  vector[N-1] alpha_rescale;
  real dm_sig;
  real pz0term;
  real dm0term;


  // peculiar redshift  Davis & Scrimgeour peculiar velocities https://arxiv.org/pdf/1405.0105.pdf
  pv = pv_sig * pv_unit;
  pv0 = pv_sig * pv0_unit;
  pz = sqrt((pv+1) ./ (1-pv))-1;
  pz0 = sqrt((pv0+1) / (1-pv0))-1;
  pz0term = fiveoverlog10 * pz0/zcmb0;

  alpha_rescale = alpha .* alpha_scale;
  dm_sig = (dm_mu+dm_tau*tan(dm_sig_unif));
  dm0term = dm_sig*dm0_unit;

  // data prediction
  for (d in 1:D){           //for each supernova pair
  // prediction for Delta
    mn[d] =  fiveoverlog10 * pz[d]/zcmb[d] - pz0term + dm_sig*dm_unit[d] - dm0term + dot_product(alpha_rescale, snparameters[d]);
  // prediction for features
    for (n in 1:N-1){
      mn[d+n*D]= snparameters[d,n];
    }
  }

  pv_unit ~ normal(0, 1);                         // peculiar velocity
  pv0_unit ~ normal(0, 1);
  dm_sig_unif ~ uniform(-pi()/4, pi()/2);         // peculiar magnitude
  dm_unit ~ normal(0, 1); 
  dm0_unit ~ normal(0, 1);

  // L_snp_cor ~ lkj_corr_cholesky(4.);            // feature covariance
  L_snp_sig_unif ~ uniform(0,pi()/2);
  for (d in 1:D){
    snparameters_alpha[d] ~ normal(0,1);       
  }

  meas ~ multi_normal_cholesky(mn, L);          // feature data
  // zcmb ~ normal(z_true,zerr);                   // redshift data
  // zcmb0 ~ normal(z0_true,zerr0);
}
