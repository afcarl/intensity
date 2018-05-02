functions {
  real p_det(real x) {
    return inv_logit(2 * (x - 3));
  }
}

transformed data {
  real mu = 5;
  real<lower=0> tau = 2;
  real<lower=0> sigma = 1;
  real<lower=0> Lambda = 100;

  int<lower=1> N_latent = poisson_rng(Lambda);
  vector[N_latent] x_obs_latent;

  int N_draw = 0;
  int<lower=0, upper=1> obs[N_latent];

  for (n in 1:N_latent) {
    real x_latent = normal_rng(mu, tau);
    x_obs_latent[n] = normal_rng(x_latent, sigma);

    obs[n] = bernoulli_rng(p_det(x_obs_latent[n]));
    if (obs[n]) N_draw = N_draw + 1;
  }
}

generated quantities {
  int<lower=0> N = N_draw;
  vector[N_draw] x_obs;

  {
    int idx = 1;
    for (n in 1:N_latent) {
      if (obs[n]) {
        x_obs[idx] = x_obs_latent[n];
        idx = idx + 1;
      }
    }
  }
}
