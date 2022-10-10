# imports
import matplotlib.pyplot as plt
from scipy.stats import beta, binom
import numpy as np
import random

# variables
NUM_FLIPS = 5
BINOM_P = 0.5
X = np.linspace(-0.01, 1.01, 1030)

# calculate prior
prior_pdf = beta.pdf(X, 1, 1)

# coin flips
n_heads = binom.rvs(NUM_FLIPS, BINOM_P)

# calc posterior
posterior_pdf = beta.pdf(X, 1 + n_heads, 1 + (NUM_FLIPS - n_heads))

# plot
fig, ax = plt.subplots(ncols=3)
fig.set_figwidth(16)
plt.setp(ax, xlim=[-0.01, 1.01], ylim=[0, 3.5])

ax[0].set_title("PRIOR")
ax[0].fill_between(X, prior_pdf, alpha=0.5)
ax[0].plot(X, prior_pdf, 'k', linewidth=1)

ax[1].set_title("DATA")
ax[1].bar(["failures", "successes"], [NUM_FLIPS - n_heads, n_heads], align="center", edgecolor="#000000", linewidth=1)
ax[1].set_xlim([-0.5, 1.5])

ax[2].set_title("POSTERIOR")
ax[2].fill_between(X, posterior_pdf, alpha=0.5)
ax[2].plot(X, posterior_pdf, 'k', linewidth=1)

plt.savefig("images/_bayes.png")
plt.show()
