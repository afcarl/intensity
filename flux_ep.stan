functions{



  // This function is the variable Poisson intensity
  real dNdz(real z, real r0, real alpha, real beta) {
    return r0 * (1.0 + z)^(alpha) * exp(-z/beta);
  }

  
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

  vector flux(vector luminosity, vector z){

    vector[num_elements(z)] z1;

    z1 = z+1;
    
    return luminosity ./ (4*pi() * z1 .* z1);

  }

    vector ep_obs(vector ep, vector z){

      return ep  ./ (1+z);
  }

  
}

data{


  int Nobs; // The number of obserations
  int Nnobs_max; // The truncation of the marginalization

  vector<lower=0>[Nobs] Fobs;
  vector<lower=0>[Nobs] epobs; 
  real sigma_F_obs;
  real sigma_ep_obs;
  real Fth; // The known threshold of measurements.
  real eth; // The known threshold of measurements. 
  real zmax; // upper bound on true values

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

  zout[1] = zmax;
}
parameters{

  // hyper parameters for the variable intensity
  real<lower=0> r0;
  real<lower=0> alpha;
  real<lower=0> beta;

  // luminosity function
  real mu;
  real<lower=0> sigma;

    // luminosity function
  real E0;
  real<lower=0> E0sigma;

  
  
  // the true redshifts truncated at z=zmax
  vector<lower=0,upper=zmax>[Nobs] ztrue;

  // non-centered approach
  vector[Nobs] Ltrue_tilde;
  vector[Nobs] Etrue_tilde;

  // latent unobserved population parameters

  // the unobserved redshifts
  vector<lower=0,upper=zmax>[Nnobs_max] znobs_true;

  // the flux parameters
  vector[Nnobs_max] Ltrue_nobs_tilde;
  vector<lower=0,upper=Fth>[Nnobs_max] flux_nobs;  

  // the ep parameters
  vector[Nnobs_max] Etrue_nobs_tilde;
  vector<lower=0,upper=eth>[Nnobs_max] ep_nobs;  

}

transformed parameters{

  // everything must be transformed via redshift




  vector<lower=0>[Nobs] Ltrue;
  vector<lower=0>[Nobs] Etrue;
  vector<lower=0>[Nnobs_max] Ltrue_nobs;
  vector<lower=0>[Nnobs_max] Etrue_nobs;
  
  vector<lower=0>[Nobs] Ftrue; 
  vector<lower=0>[Nnobs_max] Ftrue_nobs;

  vector<lower=0>[Nobs] eptrue; 
  vector<lower=0>[Nnobs_max] eptrue_nobs;

  Ltrue = exp(mu + sigma*Ltrue_tilde);
  Etrue = exp(E0 + E0sigma*Etrue_tilde);

  Ltrue_nobs = exp(mu + sigma*Ltrue_nobs_tilde);
  Etrue_nobs = exp(E0 + E0sigma*Etrue_nobs_tilde);
  

  
  Ftrue = flux(Ltrue, ztrue);
  Ftrue_nobs = flux(Ltrue_nobs, znobs_true);


  eptrue = ep_obs(Etrue, ztrue);
  eptrue_nobs = ep_obs(Etrue_nobs, znobs_true);

  
}

model{

  // setup for the integral
  real Ntrue;
  real params[3];
  real integration_result[1,1];
  real state0[1];

  
  // positive definite priors for the intensity
  r0 ~ lognormal(log(100.0), 1.0);
  alpha ~ lognormal(log(2.0), 1.0);
  beta ~ lognormal(log(2.0), 1.0);


  // luminosity function
  mu ~ normal(0.0, 3.0);
  sigma ~ normal(0.0, 0.5);

  // luminosity function
  E0 ~ normal(0.0, 10.0);
  E0sigma ~ normal(0.0, 0.5);


  params[1] = r0;
  params[2] = alpha;
  params[3] = beta;
 
  state0[1] = 0.0;

  // integrate the dN/dz to get the normalizing constant for given r0 and alpha
  integration_result = integrate_ode_rk45(N_integrand, state0, 0.0, zout, params, x_r, x_i);
  Ntrue = integration_result[1,1];

  // object level
  
  Ltrue_tilde ~ normal(0, 1);
  Etrue_tilde ~ normal(0, 1);

  //observed likelihood

  Fobs ~ lognormal(log(Ftrue), sigma_F_obs);
  epobs ~ lognormal(log(eptrue), sigma_ep_obs);

  
  // we must go thru and add on the log intensity at these points
  for (i in 1:Nobs) {
    target += log(dNdz(ztrue[i], r0, alpha, beta));
  }


  // add this to the loglikelihood
  target += -Ntrue;

  // now for the unobserved part
  
  // now marginalize out M

  // first we have to make sure any of the distributions are normalized
  // for the latent population
  
  /* target += -Nnobs_max*log(Fth); */
  /* target += -Nnobs_max*log(eth); */
  /* target += -Nnobs_max*log(zmax); */

  
  Ltrue_nobs_tilde ~ normal(0, 1);
  Etrue_nobs_tilde ~ normal(0, 1);
 

  
  {

    vector[Nnobs_max+1] marginal_term;

    // the truncation should extend far enough to cover the number of unobserved events
    for (i in 1:Nnobs_max) {
       
      // first the differential rate
      marginal_term[i+1] = log(dNdz(znobs_true[i], r0, alpha, beta));

      //now the likelihood
      marginal_term[i+1] += lognormal_lpdf(flux_nobs[i] | log(Ftrue_nobs[i]), sigma_F_obs);

      marginal_term[i+1] += lognormal_lpdf(ep_nobs[i] | log(eptrue_nobs[i]), sigma_ep_obs);

      // finally the prior corrections and factorial
      marginal_term[i+1] += -log(i);// + log(Fth) + log(zmax);

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
