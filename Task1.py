import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm

# Part 1

# List of 200 evenly spaced w0 and w1 values between -3 and 1, and -2 and 2 respectively
w0list = np.linspace(-3.0, 1.0, 200)
w1list = np.linspace(-2.0, 2.0, 200)
# Define alpha, given as 2 in task but can be changed here
alpha = 2
# Create covariance matrix alpha^(-1)*I (Equation 23)
cov_matrix = 1/alpha * np.identity(2)
# Define mu (mean) which in task is 0
mu = [0,0]

# Create distribution of the prior which is a multivariate normal distribution (Equation 23)
prior_dist = multivariate_normal(mu, cov_matrix)
# The two lines below together give all possible coordinates for w0 and w1
w0arr, w1arr = np.meshgrid(w0list, w1list)
pos = np.dstack((w0arr, w1arr))
# Probability density value of the prior at position pos
prior_pdf = prior_dist.pdf(pos)

# Contour plot the prior
plt.contour(w0arr, w1arr, prior_pdf)
plt.xlabel("w0")
plt.ylabel("w1")
plt.title(f"Prior")
plt.show()

# Part 2

# So that we get the same random values each time the program is ran
np.random.seed(0)

# Get an array [-1,-0.99,...,0.99,1]. (35) in lab instructions
x = np.arange(-1.00, 1.01, 0.01)
variance = 0.2
# Standard deviation
sigma = np.sqrt(variance)
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
    plt.title(f"Likelihood with {num_samples} samples")
    plt.show()

# Compute and plot the likelihood for different number of samples
compute_likelihood(3)
compute_likelihood(10)
compute_likelihood(20)
compute_likelihood(100)

# Part 3

# Beta is inverse of variance (sigma^(-2))
beta = 1/variance
def compute_posterior(num_samples):
    # Gets num_samples amount of random indices. replace=False so that we don't get duplicates
    indices = np.random.choice(len(x), num_samples, replace=False)
    
    # Get data subset for the random index
    x_subset = x[indices]
    t_subset = t[indices]

    # Gets Xext (phi) which becomes a (len(x_subset), 2) shaped array
    Xext = np.column_stack((np.ones(len(x_subset)), x_subset))
    # SN^(-1) (Equation 28)
    SN_inv = alpha * np.identity(2) + beta * Xext.T @ Xext
    # Takes inverse of SN^(-1) to get SN which is the posterior covariance
    SN = np.linalg.inv(SN_inv)
    # Gets the posterior mean (Equation 27)
    mN = beta * SN @ Xext.T @ t_subset
    # Creates multivariate normal distribution for the posterior (Equation 24)
    posterior_dist = multivariate_normal(mN, SN)
    # Probability density value of the posterior at position pos 
    posterior_pdf = posterior_dist.pdf(pos)
    # Contour plot the posterior
    plt.contour(w0arr, w1arr, posterior_pdf)
    plt.xlabel("w0")
    plt.ylabel("w1")
    plt.title(f"Posterior with {num_samples} samples")
    plt.show()

# Compute and plot the posterior for different number of samples
compute_posterior(3)
compute_posterior(10)
compute_posterior(20)
compute_posterior(100)