data {
  int D;                                  // Number of supernova differences

  vector[D] res;                       // Feature Data
  matrix[D, D] rescov;               // Feature Data covariance
  vector[D] mass;                         // Redshift Data
  vector[D] emass;
}

transformed data{
  cholesky_factor_cov[D] L;             // Feature Covariance Cholesky factors
  L = cholesky_decompose(rescov);
}

parameters {
  vector[D] mass_0;
  real steplow;
  real stepdelta;

  real mass_mn;
  real<lower = 0, upper = pi()/2> mass_unif;
}

model {
  vector[D] mn;                               // model vector corresponding to data
  for (d in 1:D){
    if (mass_0[d] < 10){
      mn[d] = steplow;
    } else {
      mn[d] = steplow+stepdelta;
    }
  }
  mass_0 ~ normal(mass_mn, 2*tan(mass_unif));
  res ~ multi_normal_cholesky(mn, L);          // feature data
  mass_unif ~ uniform(0,pi()/2);
  mass ~ normal(mass_0, emass);
}