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
# Probability density value of the prior at every position in pos
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
x_training = np.arange(-1.00, 1.01, 0.01)
variance = 0.2
# Standard deviation
sigma = np.sqrt(variance)
# Noise follows gaussian distribution with mu=0, sigma=sqrt(0.2). 
# Get len(x_training) random noise-values
noise_training = np.random.normal(0, sigma, len(x_training))
# The model for generating training data. (35) in lab instructions
t_training = -1.2 + 0.9 * x_training + noise_training

# Computes probability for num_samples number of random samples
def compute_likelihood(num_samples):
    # Gets num_samples amount of random indices. replace=False so that we don't get duplicates
    indices = np.random.choice(len(x_training), num_samples, replace=False)

    # Get data subset for the random index
    x_subset = x_training[indices]
    t_subset = t_training[indices]

    total_likelihood = np.ones(w0arr.shape)
    for xi, ti in zip(x_subset, t_subset):
        # Calculates the prediction formula for all data subsets
        prediction = w0arr + w1arr * xi
        # Computes likelihood of one data subset
        one_likelihood = norm.pdf(ti, prediction, sigma)
        # Computes total likelihood by multiplying all likelihoods together (Equation 17)
        total_likelihood *= one_likelihood
    return total_likelihood

# Function for contour plotting the likelihood in a subplot at desired plot index plt_idx
def plot_likelihood(num_samples, plt_idx):
    plt.subplot(2,2,plt_idx)
    # compute_likelihood(num_samples) gives total likelihood with num_samples number of samples
    plt.contour(w0arr, w1arr, compute_likelihood(num_samples))
    plt.xlabel("w0")
    plt.ylabel("w1")
    plt.title(f"Likelihood with {num_samples} samples")

# Plot 4 different likelihoods with varying num_samples in subplots of the same figure
plt.figure()
plot_likelihood(3,1)
plot_likelihood(10,2)
plot_likelihood(20,3)
plot_likelihood(100,4)
plt.show()


# Part 3

# Beta is inverse of variance (sigma^(-2))
beta = 1/variance
def compute_posterior(num_samples):
    # Gets num_samples amount of random indices. replace=False so that we don't get duplicates
    indices = np.random.choice(len(x_training), num_samples, replace=False)
    
    # Get data subset for the random index
    x_subset = x_training[indices]
    t_subset = t_training[indices]

    # Gets x_ext (phi) which becomes a (len(x_subset), 2) shaped array
    x_ext = np.column_stack((np.ones(len(x_subset)), x_subset))
    # SN^(-1) (Equation 28)
    SN_inv = alpha * np.identity(2) + beta * x_ext.T @ x_ext
    # Takes inverse of SN^(-1) to get SN which is the posterior covariance
    SN = np.linalg.inv(SN_inv)
    # Gets the posterior mean (Equation 27)
    mN = beta * SN @ x_ext.T @ t_subset
    # Creates multivariate normal distribution for the posterior (Equation 24)
    posterior_dist = multivariate_normal(mN, SN)
    return posterior_dist, x_subset, t_subset, SN, mN

# Function for plotting the posterior in a subplot at the desired plot index plot_idx
def plot_posterior(num_samples, plot_idx):
    posterior_dist, _, _, _, _ = compute_posterior(num_samples)
    # Gets probability density value for every position in pos
    posterior_pdf = posterior_dist.pdf(pos)

    # Contour plot the posterior
    plt.subplot(2, 2, plot_idx)
    plt.contour(w0arr, w1arr, posterior_pdf)
    plt.xlabel("w0")
    plt.ylabel("w1")
    plt.title(f"Posterior with {num_samples} samples")

# Plot 4 different posteriors with varying num_samples in different subplots of the same figure
plt.figure()
plot_posterior(3,1)
plot_posterior(10,2)
plot_posterior(20,3)
plot_posterior(100,4)
plt.show()


# Part 4

# Get the negative x-values in x = [-1.5, -1.4,..., -1.1, 1.1,...,1.5] (36)
x_test_left = np.arange(-1.5, -1.0, 0.1)
# Get the positive x-values in x = [-1.5, -1.4,..., -1.1, 1.1,...,1.5] (36)
x_test_right = np.arange(1.1, 1.6, 0.1)
# Combine negative and positive x-values to get full x = [-1.5, -1.4,..., -1.1, 1.1,...,1.5] (36)
x_test = np.concatenate((x_test_left, x_test_right))
# Get x-values for interval [-1.5,1.5] so that we can plot the lines for the entire interval
x_full = np.concatenate((x_test_left, x_training, x_test_right))
# Noise follows gaussian distribution with mu=0, sigma=sqrt(0.2).
# Get len(x_test) random noise-values
noise_test = np.random.normal(0, sigma, len(x_test))
# Model for generating test data (36)
t_test = -1.2 + 0.9 * x_test + noise_test

# Draws 5 model samples for weight parameters from the posterior and plots corresponding lines.
# Also plots all training and testing data, with 'x'-markers for which were used in training
def plot_model_samples(num_samples, plot_idx):
    # Gets posterior distribution, x and t subset from compute_posterior() with num_samples
    posterior_dist, x_subset, t_subset, _, _ = compute_posterior(num_samples)
    
    # Gets a random value sample of 5 values from the posterior distribution
    # Something with 50% probability in the distribution will be sampled 50% of the time
    samples = posterior_dist.rvs(5)

    # Plots the linear functions y = w0 + w1 * x for each weight sample in the same subplot
    for w0, w1 in samples:
        y = w0 + w1 * x_full
        plt.subplot(2, 2, plot_idx)
        plt.plot(x_full, y)

    # Plots all training data points
    plt.scatter(x_training, t_training, label="Training data")
    # Plots all testing data points
    plt.scatter(x_test, t_test, label="Test data")
    # Plots the data points that were used in the training of the model and mark them with 'x'
    plt.scatter(x_subset, t_subset, marker='x', label="Training subset")
    plt.title(f"Posterior Samples ({num_samples} samples)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()

# Plot 4 different model samples with varying num_samples in different subplots of the same figure
plt.figure()
plot_model_samples(3,1)
plot_model_samples(10,2)
plot_model_samples(20,3)
plot_model_samples(100,4)
plt.show()


# Part 5

# Plot predictions (mean and standard deviation) for each test point (in x_test)
def plot_predictions(num_samples, plot_idx):
    # Get posterior distribution and its parameters mN and SN
    # also get which training points were used in the run
    _, x_subset, t_subset, SN, mN = compute_posterior(num_samples)
    # Will store one predicted mean per test point
    means = []
    # Will store one predicted standard deviation per test point
    std_dev = []
    for x in x_test:
        # Build x_ext = [1,x] for the current test point
        x_ext = np.array([1,x])
        # Compute predictive mean (Equation 33)
        # This is the model's best guess for t at this x
        mean_pred = mN.T @ x_ext
        # Compute predictive variance (Equation 34)
        # 1/beta is the irreducible data noise, while (x_ext.T @ SN @ x_ext) shrinks with more data
        variance_pred = 1/beta + x_ext.T @ SN @ x_ext
        # Append predictive mean for this test point
        means.append(mean_pred)
        # Convert predictive variance to standard deviation and append
        std_dev.append(np.sqrt(variance_pred))

    # Plot in one 2x2 figure, where the subplot is decided by plot_idx
    plt.subplot(2, 2, plot_idx)
    plt.errorbar(x_test, means, std_dev, label="Predictive mean +- standard deviation")
    plt.scatter(x_training, t_training, label="Training data")
    # Mark data used in model training with 'x'
    plt.scatter(x_subset, t_subset, marker='x', label="Training subset")
    plt.title(f"Predictive distribution ({num_samples} samples)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()

# Plot 4 different predictions with varying num_samples in different subplots of the same figure
plt.figure()
plot_predictions(3,1)
plot_predictions(10,2)
plot_predictions(20,3)
plot_predictions(100,4)
plt.show()