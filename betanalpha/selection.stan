// Helpful discussion at https://github.com/farr/SelectionExample/blob/master/Gaussian.ipynb

functions {
  real log_p_det(real x) {
    return log_inv_logit(2 * (x - 3));
  }

  real log1m_p_det(real x) {
    return log1m_inv_logit(2 * (x - 3));
  }
}

data {
  int<lower=1> N;
  vector[N] x_obs;
}

transformed data {
  int<lower=1> N_margin = 100;
}

parameters {
  real mu;
  real<lower=0> tau;
  real<lower=0> sigma;
  real<lower=0> Lambda;

  vector[N] x_latent;

  vector[N_margin] x_tilde;
  vector[N_margin] x_tilde_latent;
}

model {
  real sum_log_prob_used = 0;
  real sum_log_prob_unused = 0;
  vector[N_margin + 1] log_prob_margin;

  mu ~ normal(0, 5);
  tau ~ normal(0, 2);
  sigma ~ normal(0, 2);
  Lambda ~ gamma(10, 0.06);

  // Measurement model for observed objects
  x_latent ~ normal(mu, tau);
  x_obs ~ normal(x_latent, sigma);

  for (n in 1:N)
    target += log_p_det(x_obs[n]);

  // (Distinguishiable) Poisson process model
  target += N * log(Lambda);
  target += - Lambda;

  // Measurement model for auxiliary objects
  sum_log_prob_unused =  normal_lpdf(x_tilde | 0, 1)
                       + normal_lpdf(x_tilde_latent | 0, 1);

  log_prob_margin[1] = sum_log_prob_unused;

  for (n in 1:N_margin) {
    real log_prob_used =  log1m_p_det(x_tilde[n])
                        + normal_lpdf(x_tilde[n] | x_tilde_latent[n], sigma)
                        + normal_lpdf(x_tilde_latent[n] | mu, tau);

    real log_prob_unused =  normal_lpdf(x_tilde[n] | 0, 1)
                          + normal_lpdf(x_tilde_latent[n] | 0, 1);

    sum_log_prob_used += log_prob_used;
    sum_log_prob_unused -= log_prob_unused;

    log_prob_margin[n + 1] =  n * log(Lambda) - lgamma(n + 1)
                            + sum_log_prob_used + sum_log_prob_unused;
  }

  target += log_sum_exp(log_prob_margin);
}

generated quantities {
  vector[N] x_obs_ppc;
  for (n in 1:N)
    x_obs_ppc[n] = normal_rng(x_latent[n], sigma);
}
