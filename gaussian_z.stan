functions {


  vector flux(vector luminosity, vector z){

    vector[num_elements(z)] z1;
    z1 = z+1;
    
    return luminosity ./ (4*pi()*z1 .* z1);

  }



}

data {

  int Nobs;
  
  vector[Nobs] Fobs;
  vector<lower=0>[Nobs] zobs;
  
  real Funcert; /* Fractional flux uncertainty */
  real zmax; /* Maximum redshift. */
  real Fth; /* Threshold flux */

  int NNobs_max;
  
}

parameters {

  /* Expected number of sources out to zmax. */
  real<lower=0> Lambda;

  /* Power law at low luminosity. */
  real mu;

  /* Turnover to exponential decay. */
  real<lower=0> sigma;

  /* True luminosity inferred for the observed systems. */
  vector<lower=0>[Nobs] Ltrue;

  /* Next parameters refer to the un-observed systems. */

  vector<lower=0>[NNobs_max] Ltrue_nobs;

  /* True redshift of (possibly) unobserved systems. */

  vector<lower=0,upper=zmax>[NNobs_max] ztrue_nobs;

  /* To be non-observed, we must have a flux smaller than the flux
     limit. */

  vector<lower=0,upper=Fth>[NNobs_max] flux_nobs;

}


transformed parameters{

  // I'm assuming we actually measure flux and sig_flux
  // so we need to have the right units
  
  vector<lower=0>[Nobs] Ftrue; 
  vector<lower=0>[NNobs_max] Ftrue_nobs;

  Ftrue = flux(Ltrue, zobs);
  
  Ftrue_nobs = flux(Ltrue_nobs, ztrue_nobs);
  
}


model {
  /* Priors. */
  Lambda ~ normal(0, 100.0);  
  mu ~ normal(0.0, 1.0);
  sigma ~ normal(0.0, 2.0);

  /* Observed systems (note that P_det == 1 for these systems, since
     their flux is above the limit). */

  Ltrue ~ lognormal(mu, sigma);

  
  Fobs ~ lognormal(log(Ftrue), Funcert);


  target += Nobs*log(Lambda);

  
  /* Non-observed systems are a mix of *physical* systems (counted by
     to the flux threshold. */
  
  Ltrue_nobs ~ lognormal(mu, sigma);

  /* Redshifts are flat */

  target += -NNobs_max*log(zmax);
  target += -NNobs_max*log(Fth);
      
  {

    vector[NNobs_max+1] log_poisson_term;
    
    for (i in 1:NNobs_max) {
            
      
      log_poisson_term[i+1] = log(Lambda) + lognormal_lpdf(flux_nobs[i] | log(Ftrue_nobs[i]), Funcert);
      log_poisson_term[i+1] += -log(i) + log(Fth);
    }
    
    log_poisson_term[1] = 0.0;
    
    log_poisson_term = cumulative_sum(log_poisson_term);
    
    target += log_sum_exp(log_poisson_term);

  }

  
  /* Poisson normalisation */
  target += -Lambda;
}
