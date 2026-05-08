# Instructions:
# https://canvas.kth.se/courses/62696/pages/maskininlarningsdelen 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm

w0list = np.linspace (-3.0, 1.0, 200)
w1list = np.linspace (-2.0, 2.0, 200)
print(f"W0list 1xN:\n{w0list.shape}")

W0arr, W1arr = np.meshgrid(w0list, w1list ) # 1xN, 1xM => 2st NxM
print(f"W0arr NxM:\n{W0arr.shape}")

pos = np.dstack((W0arr, W1arr)) # Stacks 2 NxM matrix ontop => 2xNxM
print(f"pos: 2xNxM:\n{pos.shape}")

# set your mu vector and Cov array
mu = [2.5, -0.5]
Cov = [
    [1,0],
    [0,1]
]

rv = multivariate_normal(mu, Cov)

# Wpriorpdf[i, j]
# (w0, w1) = (W0arr[i, j], W1arr[i, j])
Wpriorpdf = rv.pdf(pos) # returns probability for many arrays 2xNxM => takes each xn and ym and find prob. for [xn,ym]. f(xn,ym)=out[ym][xn]

plt.contour(W0arr, W1arr, Wpriorpdf) #3D plot coords: (W0arr, W1arr), value: Wpriorpdf[W1arr][W0arr]
plt.show()