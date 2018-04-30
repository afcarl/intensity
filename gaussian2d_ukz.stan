functions {

  vector flux(vector luminosity, vector z){

    // the luminosity to flux conversion.
    // this is just a dummy version for now
 
    vector[num_elements(z)] z1;
    // just keeps crazy things from happening for z<1
    z1 = z+1;

    return luminosity ./ (4*pi()*z1 .* z1);
  }


  vector ep_obs(vector ep, vector z){

    return ep  ./ (1+z);
  }

  
}

data {
  
  int Nobs; //number of observed objects
  int NNobs_max; // number of (truncated) unobserved objects
  
  vector[Nobs] Fobs; // observed fluxes
  vector[Nobs] epobs; // observed fluxes
  real Funcert; // homoskedastic flux error
  real Euncert; // homoskedastic flux error

  real zmax; // max volume of redshift
  real Fth; // known threshold flux
  real eth; // known threshold flux


  
}

parameters {

  // Expected number of sources out to zmax
  real<lower=0> Lambda;

  // the parameters of the luminosity function
  real mu;
  real<lower=0> sigma;


  real E0;
  real<lower=0> E0sigma;

  // the latent redshifts
  vector<lower=0,upper=zmax>[Nobs] ztrue;
  
  // True luminosity inferred for the observed systems.

  vector<lower=0>[Nobs] Ltrue;
  vector<lower=0>[Nobs] Etrue;

  
  // the unobserved likelihood

  vector<lower=0>[NNobs_max] Ltrue_nobs;
  vector<lower=0>[NNobs_max] Etrue_nobs;

  // True redshift of (possibly) unobserved systems.

  vector<lower=0,upper=zmax>[NNobs_max] ztrue_nobs;

  /* To be non-observed, we must have a flux smaller than the flux
     limit. */

  vector<lower=0,upper=Fth>[NNobs_max] flux_nobs;
  vector<lower=0,upper=eth>[NNobs_max] ep_nobs;

}


transformed parameters{

  // luminosites must be transformed into fluxes
  
  vector<lower=0>[Nobs] Ftrue; 
  vector<lower=0>[NNobs_max] Ftrue_nobs;

  vector<lower=0>[Nobs] Eptrue; 
  vector<lower=0>[NNobs_max] Eptrue_nobs;

  Ftrue = flux(Ltrue, ztrue);
  Ftrue_nobs = flux(Ltrue_nobs, ztrue_nobs);

  Eptrue = ep_obs(Etrue, ztrue);
  Eptrue_nobs = ep_obs(Etrue_nobs, ztrue_nobs);

  
}


model {
  /* Priors. */
  Lambda ~ normal(100, 100.0);
  
  mu ~ normal(0.0, 1.0);
  sigma ~ normal(0.0, .5);

  E0 ~ normal(0.0, 5.0);
  E0sigma ~ normal(0.0, 1);


  // the observed part of the likelihoood

  Ltrue ~ lognormal(mu, sigma);
  Fobs ~ lognormal(log(Ftrue), Funcert);


  Etrue ~ lognormal(E0, E0sigma);
  epobs ~ lognormal(log(Eptrue), Euncert);

  
  // poisson weighting
  target += Nobs*log(Lambda);



  //target += -Nobs*log(zmax);
  
  /* Non-observed systems are a mix of *physical* systems (counted by
     to the flux threshold. */
  
  Ltrue_nobs ~ lognormal(mu, sigma);
  Etrue_nobs ~ lognormal(E0, E0sigma);


  //target += -NNobs_max*log(zmax);
  // target += -NNobs_max*log(Fth);
  //  target += -NNobs_max*log(eth);
  //  target += -NNobs_max*log(zmax);
      
  {

    vector[NNobs_max+1] log_poisson_term;
    
    for (i in 1:NNobs_max) {
              
      log_poisson_term[i+1] = log(Lambda);

      log_poisson_term[i+1] += lognormal_lpdf(flux_nobs[i] | log(Ftrue_nobs[i]), Funcert);
      log_poisson_term[i+1] += lognormal_lpdf(ep_nobs[i] | log(Eptrue_nobs[i]), Euncert);
      
      log_poisson_term[i+1] += -log(i);
      //      log_poisson_term[i+1] += + log(Fth) + log(eth);// + log(zmax);
      
    }
    
    log_poisson_term[1] = 0.0;
    
    log_poisson_term = cumulative_sum(log_poisson_term);
    
    target += log_sum_exp(log_poisson_term);

  }
  
  /* Poisson normalisation */
  target += -Lambda;
}
