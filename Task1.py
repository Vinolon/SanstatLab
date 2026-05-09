import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm

# Part 1
w0list = np.linspace(-3.0, 1.0, 200)
w1list = np.linspace(-2.0, 2.0, 200)
cov_matrix = [
    [1/2,0],
    [0,1/2]
]
mu = [0,0]
prior_dist = multivariate_normal(mu, cov_matrix)
w0arr, w1arr = np.meshgrid(w0list, w1list)
pos = np.dstack((w0arr, w1arr))
prior_pdf = prior_dist.pdf(pos)
plt.contour(w0arr, w1arr, prior_pdf)
plt.show()

# Part 2
# So that we get the same random values each time the program is ran
np.random.seed(0)

# Get an array [-1,-0.99,...,0.99,1]. (35) in lab instructions
x = np.arange(-1.00, 1.01, 0.01)
# Standard deviation
sigma = np.sqrt(0.2)
# Noise follows gaussian distribution with mu = 0, sigma = sqrt(0.2). Get len(x) random noise-values
noise = np.random.normal(0, sigma, len(x))
# The model for generating testing data. (35) in lab instructions
t = -1.2 + 0.9 * x + noise

# Computes probability for num_samples number of random samples
def compute_likelihood(num_samples): 
    # Gets num_samples amount of random indices. replace=False so that we don't get duplicates
    indices = np.random.choice(len(x), num_samples, replace=False)

    # Get data subset for the random index
    x_subset = x[indices]
    t_subset = t[indices]

    total_likelihood = np.ones(w0arr.shape)
    for xi, ti in zip(x_subset, t_subset):
        # Calculates the prediction formula for all data subsets
        prediction = w0arr + w1arr * xi
        # Computes likelihood of one data subset
        one_likelihood = norm.pdf(ti, prediction, sigma)
        # Computes total likelihood by multiplying all likelihoods together (Equation 17)
        total_likelihood *= one_likelihood

    # Contour plot the likelihood
    plt.contour(w0arr, w1arr, total_likelihood)
    plt.xlabel("w0")
    plt.ylabel("w1")
    plt.title(f"Probability with {num_samples} samples")
    plt.show()

compute_likelihood(3)
compute_likelihood(10)
compute_likelihood(20)
compute_likelihood(100)