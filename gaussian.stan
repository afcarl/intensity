data {
  int Nobs;
  vector[Nobs] xobs; // obs
  real sigma_obs; // measurement error
  real xth; // know selection
  int NNobs_max; // truncation of the marginalization 
}

parameters {

  real<lower=0> Lambda;
  real mu_x;
  real<lower=0> sigma_x;


  // observed stuffs

  vector<lower=0>[Nobs] xobs_true;
  

  // now for the unbobserved part

  
  vector<lower=0>[NNobs_max] xnobs_true;
  
  // latent unobserved variables up to the threshold
  vector<lower=0,upper=xth>[NNobs_max] xnobs;

}




model {
  /* Priors */

  Lambda ~ normal(100.0, 100.0);

  mu_x ~ normal(0, 10);
  sigma_x ~ normal(0, 10);


  /* Observed likelihood */
  xobs ~ lognormal(log(xobs_true), sigma_obs);
  xobs_true ~ lognormal(mu_x, sigma_x);


  target += Nobs*log(Lambda);

  /* Non observed likelihood. */
  //xnobs_true ~ lognormal(mu_x, sigma_x);

// marginalize

//  target += -NNobs_max*log(xth); /* Default flat prior for xnobs */

 {
    vector[NNobs_max+1] log_poisson_term;

    for (i in 1:NNobs_max) {


	
      log_poisson_term[i+1] = log(Lambda);

	// likelihood
      log_poisson_term[i+1] += lognormal_lpdf(xnobs_true[i] | mu_x, sigma_x);
      log_poisson_term[i+1] += lognormal_lpdf(xnobs[i] | log(xnobs_true[i]), sigma_obs);

	// remove the prior on xnobs so everything integrates to 1
      //log_poisson_term[i+1] += log(xth);

      // divide by the factorial
      log_poisson_term[i+1] +=  - i*log(i);

    }

    log_poisson_term[1] = 0.0;

    log_poisson_term = cumulative_sum(log_poisson_term);

    target += log_sum_exp(log_poisson_term);
  }

  target += -Lambda;


}
 
