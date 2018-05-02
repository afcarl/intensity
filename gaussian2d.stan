
data {
  int Nobs;
  vector[Nobs] xobs;

  real sigma_obs;

  vector[Nobs] yobs;
  
  real xth;
  real yth;

  int NNobs_max;
}

parameters {

  real<lower=0> Lambda;

  real mu_x;
  real<lower=0> sigma_x;
  




  real mu_y;
  real<lower=0> sigma_y;

  // observed stuffs

  vector<lower=0>[Nobs] xobs_true;
  
  vector<lower=0>[Nobs] yobs_true;


  // now for the unbobserved part
  
  // The selection in the lower quadrant


  vector<lower=0>[NNobs_max] ynobs_true;
  vector<lower=0,upper=yth>[NNobs_max] ynobs;

  
  vector<lower=0>[NNobs_max] xnobs_true;
  vector<lower=0,upper=xth>[NNobs_max] xnobs;



  // where Y would be accepted but is rejected by X


  // x is non selected
  //  vector<lower=0>[NNobs_max] xnobs2_true;
  // vector<lower=xth,upper=10000>[NNobs_max] xnobs2;

  // y would be selected
  //  vector<lower=0>[NNobs_max] ynobs2_true;
  // vector<lower=yth,upper=10000>[NNobs_max] ynobs2;




}

model {
  /* Priors */

  Lambda ~ normal(100.0, 100.0);

  mu_x ~ normal(0, 10);
  sigma_x ~ normal(0, 10);

  mu_y ~ normal(0, 10);
  sigma_y ~ normal(0, 10);
  
  

  /* Observed likelihood */
  xobs ~ lognormal(log(xobs_true), sigma_obs);
  xobs_true ~ lognormal(mu_x, sigma_x);


  yobs ~ lognormal(log(yobs_true), sigma_obs);
  yobs_true ~ lognormal(mu_y, sigma_y);



  target += Nobs*log(Lambda);

  /* Non observed likelihood. */
  xnobs_true ~ lognormal(mu_x, sigma_x);
  ynobs_true ~ lognormal(mu_y, sigma_y);

  // xnobs2_true ~ lognormal(mu_x, sigma_x);
  //  ynobs2_true ~ lognormal(mu_y, sigma_y);


  //  xnobs3_true ~ lognormal(mu_x, sigma_x);
  // ynobs3_true ~ lognormal(mu_y, sigma_y);



  //  target += -NNobs_max*log(10000); /* Default flat prior for xnobs */
  // target += -NNobs_max*log(10000); /* Default flat prior for xnobs */

  target += -NNobs_max*log(xth); /* Default flat prior for xnobs */
  target += -NNobs_max*log(yth); /* Default flat prior for xnobs */

  //target += -NNobs_max*log(10000-xth); /* Default flat prior for xnobs */
  //target += -NNobs_max*log(10000-yth); /* Default flat prior for xnobs */




  
  {
    vector[NNobs_max+1] log_poisson_term;

    for (i in 1:NNobs_max) {


      log_poisson_term[i+1] = log(Lambda);

      log_poisson_term[i+1] += lognormal_lpdf(xnobs[i] | log(xnobs_true[i]), sigma_obs);

      log_poisson_term[i+1] += lognormal_lpdf(ynobs[i] | log(ynobs_true[i]), sigma_obs);


      //  log_poisson_term[i+1] += -lognormal_lpdf(xnobs2[i] | log(xnobs2_true[i]), sigma_obs);

      //log_poisson_term[i+1] += -lognormal_lpdf(ynobs2[i] | log(ynobs2_true[i]), sigma_obs);


      //      log_poisson_term[i+1] += lognormal_lpdf(xnobs3[i] | log(xnobs3_true[i]), sigma_obs);

      //  log_poisson_term[i+1] += lognormal_lpdf(ynobs3[i] | log(ynobs3_true[i]), sigma_obs);

      
      // log_poisson_term[i+1] += log(10000);

      //  log_poisson_term[i+1] += log(10000);

      log_poisson_term[i+1] += log(yth);

      log_poisson_term[i+1] += log(xth);

      // log_poisson_term[i+1] += log(10000-yth);

      //  log_poisson_term[i+1] += log(10000-xth);
      
      
      log_poisson_term[i+1] +=	-log(i);

    }

    log_poisson_term[1] = 0.0;

    log_poisson_term = cumulative_sum(log_poisson_term);

    target += log_sum_exp(log_poisson_term);
  }

  target += -Lambda;
}
