
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Model with no noice
n = 41
w = np.array([0, 2.5, -0.5])
x1 = np.linspace(-1.0, 1.0, n)
x2 = np.linspace(-1.0, 1.0, n)
X1, X2 = np.meshgrid(x1, x2)
t = np.ones((n,n))*w[0] + X1*w[1] + X2*w[2] 

#design_matrix = np.column_stack((np.ones(n), x1, x2**2)) # [1, x1, x2**2]
#t = design_matrix@w

for plot in range(3):
    # Adds noice
    noice_var = [0.3,0.6,1.2][plot]
    random_noice = np.random.normal(0, noice_var, size=(n,n) )
    noicy_t = t + random_noice

    # Plots
    plt.subplot(1, 3, plot+1)
    plt.contour(X1, X2, noicy_t) 
plt.show()