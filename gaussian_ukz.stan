functions {

  vector flux(vector luminosity, vector z){

    // the luminosity to flux conversion.
    // this is just a dummy version for now
 
    vector[num_elements(z)] z1;
    // just keeps crazy things from happening for z<1
    z1 = z+1;

    return luminosity ./ (4*pi()*z1 .* z1);
  }

}

data {

  
  int Nobs; //number of observed objects
  int NNobs_max; // number of (truncated) unobserved objects
  
  vector[Nobs] Fobs; // observed fluxes
  real Funcert; // homoskedastic flux error
  real zmax; // max volume of redshift
  real Fth; // known threshold flux


  
}

parameters {

  // Expected number of sources out to zmax
  real<lower=0> Lambda;

  // the parameters of the luminosity function
  real mu;
  real<lower=0> sigma;

  // the latent redshifts
  vector<lower=0,upper=zmax>[Nobs] ztrue;
  
  // True luminosity inferred for the observed systems.
  vector<lower=0>[Nobs] Ltrue;

  // the unobserved likelihood

  vector<lower=0>[NNobs_max] Ltrue_nobs;

  // True redshift of (possibly) unobserved systems.

  vector<lower=0,upper=zmax>[NNobs_max] ztrue_nobs;

  /* To be non-observed, we must have a flux smaller than the flux
     limit. */

  vector<lower=0,upper=Fth>[NNobs_max] flux_nobs;

}


transformed parameters{

  // luminosites must be transformed into fluxes
  
  vector<lower=0>[Nobs] Ftrue; 
  vector<lower=0>[NNobs_max] Ftrue_nobs;

  Ftrue = flux(Ltrue,ztrue);
  
  Ftrue_nobs = flux(Ltrue_nobs,ztrue_nobs);
  
}


model {
  /* Priors. */
  Lambda ~ normal(100, 100.0);  
  mu ~ normal(0.0, 1.0);
  sigma ~ normal(0.0, .5);


  // the observed part of the likelihoood

  Ltrue ~ lognormal(mu, sigma);
  Fobs ~ lognormal(log(Ftrue), Funcert);

  // poisson weighting
  target += Nobs*log(Lambda);



  //target += -Nobs*log(zmax);
  
  /* Non-observed systems are a mix of *physical* systems (counted by
     to the flux threshold. */
  
  Ltrue_nobs ~ lognormal(mu, sigma);

  /* Redshifts are flat */


  //target += -NNobs_max*log(zmax);
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
