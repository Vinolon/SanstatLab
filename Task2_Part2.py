import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_data(x1, x2, w, noice_var):
    X1, X2 = np.meshgrid(x1, x2)
    random_noice = np.random.normal(0, noice_var, size=(len(x2),len(x1)) )
    t = np.ones((len(x2),len(x1)))*w[0] + X1*w[1] + X2*w[2] + random_noice    
    return t

# Returns <size>% of the data 
def get_training_data(x1, x2, t, size):
    num_x1 = int(len(x1)*size//2)
    num_x2 = int(len(x2)*size//2)
    x1_train = np.concatenate((x1[:num_x1], x1[num_x1:]))
    x2_train = np.concatenate((x2[:num_x2], x2[num_x2:]))
    t_train_L = np.concatenate((t[:num_x2, :num_x1], t[num_x2:, :num_x1]))
    t_train_R = np.concatenate((t[:num_x2, :num_x1], t[num_x2:, :num_x1])) # WRONG
    t_train = np.concatenate((t_train_L, t_train_R), axis=1)
    return x1_train, x2_train, t_train

def main():
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.linspace(-1.0, 1.0, n)
    w = np.array([0, 2.5, -0.5])
    t = generate_data(x1, x2, w, noice_var)

    
    

main()

