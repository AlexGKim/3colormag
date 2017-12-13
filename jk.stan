#./gerard11 sample num_warmup=5000 num_samples=5000 data file=data.R init=init11.R output file=output11.csv refresh=1000

data {
  int D;                // Number of supernova differences
  int N;                // Number of features + Delta

  vector[D*N] meas;
  matrix[D*N, D*N] meascov;
  vector[D] zcmb;
  real zcmb0;

  int holdout;
}

transformed data{
  cholesky_factor_cov[D*N] L;
  real pv_sig;
  L = cholesky_decompose(meascov);
  pv_sig=300./3e5;
}

parameters {
  # order is EWCa, EWSi, lambdaSi, p
  matrix[D, (N-1)] snparameters;
  vector[N-1] alpha;

  vector[D] pv_unit;
  real pv0_unit;

  real<lower=0> dm_sig;
  vector[D] dm_unit;
  real dm0_unit;  

  real delta_holdout;
}

# Davis & Scrimgeour peculiar velocities https://arxiv.org/pdf/1405.0105.pdf

transformed parameters {

  vector[D] pz;
  real pz0;
  vector[D*N] mn;
  {
    vector[D] pv;
    real pv0;

    pv = pv_sig * pv_unit;
    pz = (pv+1) ./ (1-pv);
    for (d in 1:D){
      pz[d] = sqrt(pz[d])-1;
    }
    pv0 = pv_sig * pv0_unit;
    pz0 = sqrt((pv0+1)/ (1-pv0))-1;
  }

  # prediction for Delta
  for (d in 1:D){
    mn[d]= 5/log(10.)*(pz[d]/zcmb[d] - pz0/zcmb0) + dm_sig*(dm_unit[d] - dm0_unit);
    for (n in 1:N-1){
      mn[d] = mn[d]+ alpha[n] * snparameters[d,n] ; 
    }
  }

  mn[holdout] = delta_holdout;

  #prediction for features
  for (n in 1:N-1){
      for (d in 1:D){
        mn[d+n*D]= snparameters[d,n];
      }
  } 

}

model {

  pv_unit ~ normal(0, 1);
  pv0_unit ~ normal(0, 1);
  dm_unit ~ normal(0, 1); 
  dm0_unit ~ normal(0, 1);

  meas ~ multi_normal_cholesky(mn, L);

  dm_sig ~ cauchy(0.08, 0.08);

}

generated quantities {
  real residual;
  residual = delta_holdout - (5/log(10.)*(pz[holdout]/zcmb[holdout] - pz0/zcmb0) + dm_sig*(dm_unit[holdout] - dm0_unit));
  for (n in 1:N-1){
    residual = residual - alpha[n] * snparameters[holdout,n] ; 
  }
}
