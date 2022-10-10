# imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import random

# hyperparams
NUM_DRAWS = 16
NUM_POINTS = 3

# make gaussian process
kernel = RBF(length_scale_bounds=(0, 1))
X = np.linspace(0, 1, 101)
gp = GaussianProcessRegressor(kernel=kernel)
pre_gp_means = gp.predict(X.reshape(-1, 1))
pre_gp_draws = [gp.sample_y(X.reshape(-1, 1), random_state=i) for i in range(NUM_DRAWS)]

# random couple data points
rand_x = np.array([(i + random.random()) / NUM_POINTS for i in range(NUM_POINTS)])
rand_y = np.array([random.random() * 2 - 1 for i in range(NUM_POINTS)])

# update gp
gp.fit(rand_x.reshape(-1, 1), rand_y.reshape(-1, 1))
post_gp_means = gp.predict(X.reshape(-1, 1))
post_gp_draws = [gp.sample_y(X.reshape(-1, 1), random_state=i) for i in range(NUM_DRAWS)]

# plot
fig, ax = plt.subplots(ncols=2)
fig.set_figwidth(12)

ax[0].set_title("PRIOR")
for i in range(NUM_DRAWS):
	ax[0].plot(X, pre_gp_draws[i].reshape(-1), 'b', linewidth=0.5)
ax[0].plot(X, pre_gp_means, 'k--', linewidth=2)

ax[1].set_title("POSTERIOR")
for i in range(NUM_DRAWS):
	ax[1].plot(X, post_gp_draws[i].reshape(-1), 'b', linewidth=0.5)
ax[1].plot(X, post_gp_means, 'k--', linewidth=2)
ax[1].plot(rand_x, rand_y, 'ko', ms=5)

plt.savefig("images/_gp.png")
plt.show()