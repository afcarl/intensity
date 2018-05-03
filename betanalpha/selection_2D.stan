// Helpful discussion at https://github.com/farr/SelectionExample/blob/master/Gaussian.ipynb

functions {
  real log_p_det(real x, real y) {
    return log_inv_logit(2 * (x + y - 3));
  }

  real log1m_p_det(real x, real y) {
    return log1m_inv_logit(2 * (x + y - 3));
  }
}

data {
  int<lower=1> N;
  vector[N] x_obs;
  vector[N] y_obs;
}

transformed data {
  int<lower=1> N_margin = 100;
  real<lower=0> sigma_x =1.;
  real<lower=0> sigma_y = 0.75;
  
}

parameters {
  real mu_x;
  real<lower=0> tau_x;
  //  real<lower=0> sigma_x;

  real mu_y;
  real<lower=0> tau_y;
  //  real<lower=0> sigma_y;

  real<lower=0> Lambda;

  vector[N] x_latent;

  vector[N_margin] x_tilde;
  vector[N_margin] x_tilde_latent;

  vector[N] y_latent;

  vector[N_margin] y_tilde;
  vector[N_margin] y_tilde_latent;
}

model {
  //vector[N_margin + 1] log_prob_tilde;
  real sum_log_prob_tilde;
  vector[N_margin + 1] log_prob_margin;

  mu_x ~ normal(0, 5);
  tau_x ~ normal(0, 2);
  //  sigma_x ~ normal(0, 2);

  mu_y ~ normal(0, 5);
  tau_y ~ normal(0, 2);
  // sigma_y ~ normal(0, 2);

  Lambda ~ gamma(10, 0.06);

  // Measurement model for observed objects
  x_latent ~ normal(mu_x, tau_x);
  x_obs ~ normal(x_latent, sigma_x);

  y_latent ~ normal(mu_y, tau_y);
  y_obs ~ normal(y_latent, sigma_y);

  for (n in 1:N)
    target += log_p_det(x_obs[n], y_obs[n]);

  // (Distinguishiable) Poisson process model
  target += N * log(Lambda);
  target += - Lambda;

  // Measurement model for auxiliary objects
  x_tilde_latent ~ normal(mu_x, tau_x);
  y_tilde_latent ~ normal(mu_y, tau_y);

  // Arbitrary prior on x_tilde to ensure identification
  // target += to ensure proper normalization
  target += normal_lpdf(x_tilde | 0, 1);
  target += normal_lpdf(y_tilde | 0, 1);

  log_prob_margin[1] = 0;
  sum_log_prob_tilde = log_prob_margin[1];
  //  log_prob_margin[1] = log_prob_margin[1];

  for (n in 1:N_margin) {
    sum_log_prob_tilde +=  log1m_p_det(x_tilde[n], y_tilde[n])
                           + normal_lpdf(x_tilde[n] | x_tilde_latent[n], sigma_x)
      //                     + normal_lpdf(x_tilde_latent[n] | mu_x, tau_x)
                           - normal_lpdf(x_tilde[n] | 0, 1)
                           + normal_lpdf(y_tilde[n] | y_tilde_latent[n], sigma_y)
      //                   + normal_lpdf(y_tilde_latent[n] | mu_y, tau_y)
                           - normal_lpdf(y_tilde[n] | 0, 1);

    //    sum_log_prob_tilde += log_prob_tilde[n + 1];
    log_prob_margin[n + 1] =  n * log(Lambda) - lgamma(n + 1)
                            + sum_log_prob_tilde;
  }

  target += log_sum_exp(log_prob_margin);
}

generated quantities {
  vector[N] x_obs_ppc;
  vector[N] y_obs_ppc;
  for (n in 1:N) {
    x_obs_ppc[n] = normal_rng(x_latent[n], sigma_x);
    y_obs_ppc[n] = normal_rng(y_latent[n], sigma_y);
  }
}
