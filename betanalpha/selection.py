############################################################
# Initial setup
############################################################

import pystan
import stan_utility
import matplotlib
import matplotlib.pyplot as plot
import numpy

light="#DCBCBC"
light_highlight="#C79999"
mid="#B97C7C"
mid_highlight="#A25050"
dark="#8F2727"
dark_highlight="#7C0000"
green="#00FF00"

############################################################
#
# One-dimensional
#
############################################################

############################################################
# Create data
############################################################

model = stan_utility.compile_model('generate_data.stan')
fit = model.sampling(seed=194838, algorithm='Fixed_param', iter=1, chains=1)

data = dict(N = fit.extract()['N'].astype(numpy.int64),
            x_obs = fit.extract()['x_obs'][0,:])

pystan.stan_rdump(data, 'selection.data.R')

############################################################
# Fit model
############################################################

data = pystan.read_rdump('selection.data.R')

model = stan_utility.compile_model('selection.stan')
fit = model.sampling(data=data, chains=4, seed=4938483,
                     control=dict(adapt_delta=0.9, max_treedepth=12))

# Check diagnostics
stan_utility.check_all_diagnostics(fit)

# Default visual summaries
params = fit.extract()

# Plot marginal posteriors
f, axarr = plot.subplots(2, 2)
for a in axarr[0,:]:
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('none')
for a in axarr[1,:]:
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('none')

axarr[0, 0].set_title("mu")
axarr[0, 0].hist(params['mu'], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 0].axvline(x=5, linewidth=2, color=light)

axarr[0, 1].set_title("tau")
axarr[0, 1].hist(params['tau'], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 1].axvline(x=2, linewidth=2, color=light)

axarr[1, 0].set_title("sigma")
axarr[1, 0].hist(params['sigma'], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 0].axvline(x=1, linewidth=2, color=light)

axarr[1, 1].set_title("Lambda")
axarr[1, 1].hist(params['Lambda'], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 1].axvline(x=100, linewidth=2, color=light)

plot.show()

plot.scatter(params['sigma'], params['tau'], color = mid_highlight, alpha=0.05)
plot.show()

############################################################
#
# Two-dimensional
#
############################################################

############################################################
# Create data
############################################################

model = stan_utility.compile_model('generate_data_2D.stan')
fit = model.sampling(seed=194838, algorithm='Fixed_param', iter=1, chains=1)

data = dict(N = fit.extract()['N'].astype(numpy.int64),
            x_obs = fit.extract()['x_obs'][0,:],
            y_obs = fit.extract()['y_obs'][0,:])

pystan.stan_rdump(data, 'selection_2D.data.R')

############################################################
# Fit model
############################################################

data = pystan.read_rdump('selection_2D.data.R')

model = stan_utility.compile_model('selection_2D.stan')
fit = model.sampling(data=data, chains=4, seed=4938483,
                     control=dict(adapt_delta=0.9, max_treedepth=12))

# Check diagnostics
stan_utility.check_all_diagnostics(fit)

# Default visual summaries
params = fit.extract()

# Plot marginal posteriors
f, axarr = plot.subplots(2, 4)
for a in axarr[0,:]:
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('none')
for a in axarr[1,:]:
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('none')

axarr[0, 0].set_title("mu_x")
axarr[0, 0].hist(params['mu_x'], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 0].axvline(x=5, linewidth=2, color=light)

axarr[0, 1].set_title("tau_x")
axarr[0, 1].hist(params['tau_x'], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 1].axvline(x=2, linewidth=2, color=light)

axarr[0, 2].set_title("sigma_x")
axarr[0, 2].hist(params['sigma_x'], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 2].axvline(x=1, linewidth=2, color=light)

axarr[0, 3].set_title("mu_y")
axarr[0, 3].hist(params['mu_y'], bins = 25, color = dark, ec = dark_highlight)
axarr[0, 3].axvline(x=-3, linewidth=2, color=light)

axarr[1, 0].set_title("tau_y")
axarr[1, 0].hist(params['tau_y'], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 0].axvline(x=3, linewidth=2, color=light)

axarr[1, 1].set_title("sigma_y")
axarr[1, 1].hist(params['sigma_y'], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 1].axvline(x=0.75, linewidth=2, color=light)

axarr[1, 2].set_title("Lambda")
axarr[1, 2].hist(params['Lambda'], bins = 25, color = dark, ec = dark_highlight)
axarr[1, 2].axvline(x=100, linewidth=2, color=light)

plot.show()
