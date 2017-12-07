#./gerard11 sample num_warmup=5000 num_samples=5000 data file=data.R init=init11.R output file=output11.csv refresh=1000

data {
  int D;                // Number of supernova differences
  int N;                // Number of features + Delta

  vector[D*N] meas;
  matrix[D*N, D*N] meascov;
  vector[D] zcmb;
  real zcmb0;
}

transformed data{
  cholesky_factor_cov[D*N] L;
  L = cholesky_decompose(meascov);
}

parameters {
  # order is EWCa, EWSi, lambdaSi, p
  matrix[D, (N-1)] snparameters;
  vector[N-1] alpha;

  real<lower=0> pv_sig;
  vector[D] pv_unit;
  real pv0_unit;
}

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
    mn[d]= 5/log(10.)*(pz[d]/zcmb[d] - pz0/zcmb0);
    for (n in 1:N-1){
      mn[d] = mn[d]+ alpha[n] * snparameters[d,n] ; 
    }
  }

  #prediction for features
  for (n in 1:N-1){
      for (d in 1:D){
        mn[d+n*D]= snparameters[d,n];
      }
  } 
}

model {
  target += normal_lpdf(pv_unit| 0, 1);
  target += normal_lpdf(pv0_unit| 0, 1);
  target += -(D+1)*log(pv_sig);
  meas ~ multi_normal_cholesky(mn, L);

  pv_sig ~ cauchy(300./1e5, 300./1e5);
}
