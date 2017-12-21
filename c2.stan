data {
  int D;                                  // Number of supernova differences
  int N;                                  // Number of features + Delta

  vector[D*N] meas;                       // Feature Data
  matrix[D*N, D*N] meascov;               // Feature Data covariance
  vector[D] zcmb;                         // Redshift Data
  real zcmb0;
  vector[D] zerr;                         // Redshift Uncertainty
  real zerr0;

  vector[N-1] cauchy_tau;                 // tau of the cauchy prior of distributions of parameters
  vector[N-1] alpha_scale;                // scaling to alpha to try to get sigma around 1
}

transformed data{
  cholesky_factor_cov[D*N] L;             // Feature Covariance Cholesky factors
  real pv_sig;                            // hand-specified peculiar velocity stanadard deviation
  real fiveoverlog10;

  L = cholesky_decompose(meascov);
  pv_sig=300./3e5;
  fiveoverlog10 = 5/log(10.);
}

parameters {
  vector[N-1] alpha;                      // feature cofactors

  vector[D] pv_unit;                      // peculiar velocity
  real pv0_unit;

  real<lower=-pi()/4, upper=pi()/2> dm_sig_unif;   // peculiar magnitude
  vector[D] dm_unit;
  real dm0_unit; 

  vector[D] z_true;                       // CmB redshift
  real z0_true;

  // vector[N-1] snparameters_alpha[D];      // SN parameters in linear independent form
  // cholesky_factor_corr[N-1] L_snp_cor;    // correlation matrix of SN parameters
  // vector<lower=0, upper=pi()/2>[N-1] L_snp_sig_unif;         // sigma of the covariance matrix for SN parameters
  // vector[N-1] snp_mn;                     // distribution of parameters mn

  vector[N-1] snparameters[D];
}

// transformed parameters{
//   vector[N-1] snparameters[D];      // SN parameters covariance term

//   {
//     matrix[N-1, N-1] L_snp_cov;                   // covariance matrix of SN parameters
//     // feature covariance
//     L_snp_cov = diag_pre_multiply(cauchy_tau .* tan(L_snp_sig_unif), L_snp_cor);
//     for (d in 1:D){           //for each supernova pair
//       snparameters[d] =  snp_mn + L_snp_cov * snparameters_alpha[d];
//     }
//   }
// }

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
  pz0term = fiveoverlog10 * pz0/z0_true;

  alpha_rescale = alpha .* alpha_scale;
  dm_sig = (0.08+0.08*tan(dm_sig_unif));
  dm0term = dm_sig*dm0_unit;

  // data prediction
  for (d in 1:D){           //for each supernova pair
  // prediction for Delta
    // mn[d] =  5/log(10.)*(pz[d]/z_true[d] - pz0/z0_true) + dm_sig*(dm_unit[d] - dm0_unit) + dot_product(alpha_rescale, snparameters[d]);
    mn[d] =  fiveoverlog10 * pz[d]/z_true[d] - pz0term + dm_sig*dm_unit[d] - dm0term + dot_product(alpha_rescale, snparameters[d]);
  // prediction for features
    for (n in 1:N-1){
      mn[d+n*D]= snparameters[d,n];
    }
  }

  pv_unit ~ normal(0, 1);                         // peculiar velocity
  pv0_unit ~ normal(0, 1);
  // dm_sig ~ cauchy(0.08, 0.08);                  // peculiar magnitude
  dm_sig_unif ~ uniform(-pi()/4, pi()/2);
  dm_unit ~ normal(0, 1); 
  dm0_unit ~ normal(0, 1);

  // L_snp_cor ~ lkj_corr_cholesky(2.);            // feature covariance
  // L_snp_sig_unif ~ uniform(0,pi()/2);
  // for (d in 1:D){
  //   snparameters_alpha[d] ~ normal(0,1);       
  // }

  meas ~ multi_normal_cholesky(mn, L);          // feature data
  zcmb ~ normal(z_true,zerr);                   // redshift data
  zcmb0 ~ normal(z0_true,zerr0);
}
