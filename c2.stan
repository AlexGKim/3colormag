// ./gerard11 sample num_warmup=5000 num_samples=5000 data file=data.R init=init11.R output file=output11.csv refresh=1000

data {
  int D;                                  // Number of supernova differences
  int N;                                  // Number of features + Delta

  vector[D*N] meas;                       // Feature Data
  matrix[D*N, D*N] meascov;               // Feature Data covariance
  vector[D] zcmb;                         // Redshift Data
  real zcmb0;
  vector[D] zerr;                         // Redshift Uncertainty
  real zerr0;
}

transformed data{
  cholesky_factor_cov[D*N] L;             // Feature Covariance Cholesky factors
  real pv_sig;                            // hand-specified peculiar velocity stanadard deviation
  vector[N-1] cauchyp;                    // hand-specified Cauchy parameter for feature prior

  L = cholesky_decompose(meascov);
  pv_sig=300./3e5;
  cauchyp[1]=100.; cauchyp[2]=25; cauchyp[3]=70; cauchyp[4]=4; cauchyp[5]=0.1;
}

parameters {
  vector[N-1] snparameters_zero[D];       // SN parameters pre-covariance
  vector[N-1] snparameters[D];            // SN parameters post-covariance
  vector[N-1] alpha;                      // feature cofactors

  vector[D] pv_unit;                      // peculiar velocity
  real pv0_unit;

  real<lower=0> dm_sig;                   // peculiar magnitude
  vector[D] dm_unit;
  real dm0_unit; 

  // Changes
  vector[D] z_true;                       // CmB redshift
  real z0_true;

  cholesky_factor_corr[N-1] L_snp_cor;    // correlation matrix of SN parameters
  vector<lower=0>[N-1] L_snp_sig;         // sigma of the covariance matrix for SN parameters
  vector[N-1] snp_mn;                     // distribution of parameters mn
  vector<lower=0>[N-1] snp_sig;           // distribution or parameters sig
}


// transformed parameters {

//   // vector[D] pz;                           // preculiar redshift
//   // real pz0;
//   // vector[D*N] mn;                         // model vector corresponding to data
//   cholesky_factor_cov[N-1] L_snp_cov;     // covariance matrix of SN parameters

//   L_snp_cov = diag_pre_multiply(L_snp_sig, L_snp_cor);

//   // {
//   //   vector[D] pv;
//   //   real pv0;

//   //   pv = pv_sig * pv_unit;
//   //   pz = (pv+1) ./ (1-pv);
//   //   for (d in 1:D){
//   //     pz[d] = sqrt(pz[d])-1;
//   //   }
//   //   pv0 = pv_sig * pv0_unit;
//   //   pz0 = sqrt((pv0+1)/ (1-pv0))-1;
//   // }

//   // // prediction for Delta
//   // for (d in 1:D){
//   //   mn[d]= 5/log(10.)*(pz[d]/z_true[d] - pz0/z0_true) + dm_sig*(dm_unit[d] - dm0_unit);
//   //   for (n in 1:N-1){
//   //     mn[d] = mn[d]+ alpha[n] * snparameters[d,n] ; 
//   //   }
//   // }

//   // // prediction for features
//   // for (n in 1:N-1){
//   //     for (d in 1:D){
//   //       mn[d+n*D]= snparameters[d,n];
//   //     }
//   // }
// }

model {

  vector[D*N] mn;                               // model vector corresponding to data
  vector[D] pz;                                 // peculiar redshift
  real pz0;
  vector[D] pv;                                 // peculiar velocity
  real pv0;
  matrix[N-1, N-1] L_snp_cov;                   // covariance matrix of SN parameters


  // peculiar redshift  Davis & Scrimgeour peculiar velocities https://arxiv.org/pdf/1405.0105.pdf
  pv = pv_sig * pv_unit;
  pz = (pv+1) ./ (1-pv);
  for (d in 1:D){
    pz[d] = sqrt(pz[d])-1;
  }
  pv0 = pv_sig * pv0_unit;
  pz0 = sqrt((pv0+1)/ (1-pv0))-1;

  // prediction for Delta
  for (d in 1:D){
    mn[d]= 5/log(10.)*(pz[d]/z_true[d] - pz0/z0_true) + dm_sig*(dm_unit[d] - dm0_unit);
    for (n in 1:N-1){
      mn[d] = mn[d]+ alpha[n] * snparameters[d,n] ; 
    }
  }

  // prediction for features
  for (n in 1:N-1){
      for (d in 1:D){
        mn[d+n*D]= snparameters[d,n];
      }
  }

  // feature covariance
  L_snp_cov = diag_pre_multiply(L_snp_sig, L_snp_cor);

  pv_unit ~ normal(0, 1);                       // peculiar velocity
  pv0_unit ~ normal(0, 1);
  dm_sig ~ cauchy(0.08, 0.08);                  // peculiar magnitude
  dm_unit ~ normal(0, 1); 
  dm0_unit ~ normal(0, 1);

  meas ~ multi_normal_cholesky(mn, L);          // feature data

  zcmb ~ normal(z_true,zerr);                   // redshift data
  zcmb0 ~ normal(z0_true,zerr0);

  L_snp_cor ~ lkj_corr_cholesky(2.);            // feature covariance
  L_snp_sig ~ cauchy(0,cauchyp);

  snp_sig ~ cauchy(0, cauchyp);                 // feature distribution
  for (d in 1:D){
    snparameters_zero[d] ~ normal(snp_mn,snp_sig);
    snparameters[d] ~ multi_normal_cholesky(snparameters_zero[d],L_snp_cov);
  }
}
