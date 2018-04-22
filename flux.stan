functions{



  // This function is the variable Poisson intensity
  real dNdz(real z, real r0, real alpha, real beta) {
    return r0 * (1.0 + z)^(alpha) * exp(-z/beta);
  }

  /* real schechter_lpdf(vector L, real alpha, real Lstar) { */
  /*   return sum(alpha*log(L) - L/Lstar - (alpha+1)*log(Lstar) - lgamma(alpha+1)); */
  /* } */
  
  // While we can do the integral in this case analytically,
  // I want to go ahead and allow for the ability to perform the
  // integral as the model expands
  real[] N_integrand(real z, real[] state, real[] params, real[] x_r, int[] x_i) {

    real r0;
    real alpha;
    real beta;
    real dstatedz[1];
    
    r0 = params[1];
    alpha = params[2];
    beta = params[3];
    
    dstatedz[1] = dNdz(z, r0, alpha, beta);
    
    return dstatedz;
  }

  
  
}

data{
  int Nobs; // The number of obserations
  vector<lower=0>[Nobs] zobs; // \hat{z}: the observed redshifts
  vector<lower=0>[Nobs] Fobs;
  

  real sigma_F_obs;

  real sigma_z_obs; // The homoskedastic measurement error


  
  real Fth; // The known threshold of measurements.
  int Nnobs_max; // The truncation of the marginalization

  real true_upper; // upper bound on true values
  real meas_upper; // upper bound on measured values

  int nmodel; /* The number of points at which the model should be
		 computed and stored (for convenient plotting
		 later). */
  real zs_model[nmodel]; /* The redshift of those points. */
}

transformed data {

  // technical varaibales required for the integration
  real x_r[0];
  int x_i[0];
  real zout[1];

  zout[1] = true_upper;
}
parameters{

  // hyper parameters for the variable intensity
  real<lower=0> r0;
  real<lower=0> alpha;
  real<lower=0> beta;

  // luminosity function
  real mu;
  real<lower=0> sigma;

  
  // the true redshifts truncated at z=10
  vector<lower=0,upper=true_upper>[Nobs] ztrue;
  vector<lower=0>[Nobs] Ltrue;


  // The unobserved true and measured redshifts
  // The idea here is that the selection is simulated
  // because the observed data can only come from above
  // the redshift threshold.


  vector<lower=0>[Nnobs_max] Ltrue_nobs;


  vector<lower=0,upper=Fth>[Nnobs_max] flux_nobs;  
  vector<lower=0,upper=true_upper>[Nnobs_max] znobs_true;
  vector<lower=0,upper=meas_upper>[Nnobs_max] znobs;
  
}

transformed parameters{

    
  vector<lower=0>[Nobs] Ftrue; 
  vector<lower=0>[Nnobs_max] Ftrue_nobs;

  Ftrue = Ltrue ./ (4*pi() * ztrue .* ztrue);
  Ftrue_nobs = Ltrue_nobs ./ (4*pi() * znobs_true .* znobs_true);



}

model{

  // setup for the integral
  real Ntrue;
  real params[3];
  real integration_result[1,1];
  real state0[1];


  
  // positive definite priors for the intensity
  r0 ~ lognormal(log(10.0), 1.0);
  alpha ~ lognormal(log(2.0), 1.0);
  beta ~ lognormal(log(2.0), 1.0);

  mu ~ normal(0.0, 1.0);
  sigma ~ normal(0.0, 2.0);
  
  Ltrue ~ lognormal(mu, sigma);

  

  params[1] = r0;
  params[2] = alpha;
  params[3] = beta;
 
  state0[1] = 0.0;

  // integrate the dN/dz to get the normalizing constant for given r0 and alpha
  integration_result = integrate_ode_rk45(N_integrand, state0, 0.0, zout, params, x_r, x_i);
  Ntrue = integration_result[1,1];

  // add this to the loglikelihood
  target += -Ntrue;  


  /* Observed likelihood */
  zobs ~ lognormal(log(ztrue), sigma_z_obs);
  Fobs ~ lognormal(log(Ftrue), sigma_F_obs);

  
  // we must go thru and add on the log intensity at these points
  for (i in 1:Nobs) {
    target += log(dNdz(ztrue[i], r0, alpha, beta));
  }

  
  // now marginalize out M

  target += -Nnobs_max*log(meas_upper);

  target += -Nnobs_max*log(true_upper);

  target += -Nnobs_max*log(Fth);


  Ltrue_nobs ~ lognormal(mu, sigma);

  
  {
    vector[Nnobs_max+1] marginal_term;

    // the truncation should extend far enough to cover the number of unobserved events
    for (i in 1:Nnobs_max) {

      
      // first the differential rate
      marginal_term[i+1] = log(dNdz(znobs_true[i], r0, alpha, beta));

      //now the likelihood
      marginal_term[i+1] += lognormal_lpdf(znobs[i] | log(znobs_true[i]), sigma_z_obs) + lognormal_lpdf(flux_nobs[i] | log(Ftrue_nobs[i]), sigma_F_obs);

      // finally the prior corrections
      marginal_term[i+1] += -log(i) +  log(meas_upper) + log(Fth);

    }

    // tack on the first term
    marginal_term[1] = 0.0;
    
    marginal_term = cumulative_sum(marginal_term);

    // add on to the likelihood
    target += log_sum_exp(marginal_term);
  }

  
}


generated quantities{

  real dNdz_model[nmodel];

  for (i in 1:nmodel) {
    dNdz_model[i] = dNdz(zs_model[i], r0, alpha, beta);
  }

}
