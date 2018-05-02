functions {
  real p_det(real x, real y ) {
    return inv_logit(2 * (x + y - 3));

  }
}

transformed data {
  real mu_x = 5;
  real<lower=0> tau_x = 2;
  real<lower=0> sigma_x = 1;

  real mu_y = -3;
  real<lower=0> tau_y = 3;
  real<lower=0> sigma_y = 0.75;

  real<lower=0> Lambda = 100;

  int<lower=1> N_latent = poisson_rng(Lambda);
  vector[N_latent] x_obs_latent;
  vector[N_latent] y_obs_latent;

  int N_draw = 0;
  int<lower=0, upper=1> obs[N_latent];

  for (n in 1:N_latent) {
    real x_latent = normal_rng(mu_x, tau_x);
    real y_latent = normal_rng(mu_y, tau_y);

    x_obs_latent[n] = normal_rng(x_latent, sigma_x);
    y_obs_latent[n] = normal_rng(y_latent, sigma_y);

    obs[n] = bernoulli_rng(p_det(x_obs_latent[n], y_obs_latent[n]));
    if (obs[n]) N_draw = N_draw + 1;
  }
}

generated quantities {
  int<lower=0> N = N_draw;
  vector[N_draw] x_obs;
  vector[N_draw] y_obs;

  {
    int idx = 1;
    for (n in 1:N_latent) {
      if (obs[n]) {
        x_obs[idx] = x_obs_latent[n];
        y_obs[idx] = y_obs_latent[n];
        idx = idx + 1;
      }
    }
  }
}
