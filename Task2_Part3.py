import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_data(x1, x2, w, noice_var):
    X1, X2 = np.meshgrid(x1, x2)
    random_noice = np.random.normal(0, noice_var, size=(len(x2),len(x1)) )
    t = np.ones((len(x2), len(x1)))*w[0] + (X1**2)*w[1] + (X2**3)*w[2] + random_noice    
    return x1, x2, t

# Returns <size>% of the data 
def get_test_data(data, size):
    x1, x2, t = data
    num_x1 = int(len(x1)*size//2) # get outside index, exluding index
    num_x2 = int(len(x2)*size//2)
    x1_t = np.concatenate((x1[:num_x1], x1[-num_x1:]))
    x2_t = np.concatenate((x2[:num_x2], x2[-num_x2:]))
    t_t_T = np.concatenate((t[:num_x2, :num_x1], t[-num_x2:, :num_x1]))
    t_t_B = np.concatenate((t[:num_x2, -num_x1:], t[-num_x2:, -num_x1:])) # ROW
    t_t = np.concatenate((t_t_T, t_t_B), axis=1)
    return x1_t, x2_t, t_t

def get_training_data(data, size):
    x1, x2, t = data
    num_x1 = int(len(x2)*(1-size)//2) # get between index, includeing index
    num_x2 = int(len(x2)*(1-size)//2)
    x1_t = x1[num_x1 : -num_x1]
    x2_t = x2[num_x2 : -num_x2]
    t_t = t[num_x2:-num_x2, num_x1:-num_x1]
    return x1_t, x2_t, t_t

def get_biased_test_data(data, size):
    x1, x2, t = data
    num_x1 = int(len(x1)*size) # get outside index, exluding index
    num_x2 = int(len(x2)*size)
    x1_t = x1[-num_x1:]
    x2_t = x2[-num_x2:]
    t_t = t[-num_x1:, -num_x2:]
    return x1_t, x2_t, t_t

def get_biased_training_data(data, size):
    x1, x2, t = data
    num_x1 = int(len(x1)*size) # get outside index, exluding index
    num_x2 = int(len(x2)*size)
    x1_t = x1[:-num_x1]
    x2_t = x2[:-num_x2]
    t_t = t[:-num_x1, :-num_x2]
    return x1_t, x2_t, t_t

# Splits the data into <test_size>% test data and the rest trainging data
def split_data(data, test_size):
    test_data = get_test_data(data, train_size)
    train_data = get_training_data(data, 1-train_size)
    return test_data, train_data

# Splits the data into <test_size>% test data and the rest trainging data
# Uses a biased splitting aproach
def split_data_biased(data, test_size):
    test_data = get_biased_test_data(data, train_size)
    train_data = get_biased_training_data(data, 1-train_size)
    return test_data, train_data
    
# Adds noice ~ N(0, var) to a matrix mat
# Mutates mat, Returns nothing
def add_noice(mat, var):
    random_noice = np.random.normal(0, var, size=(mat.shape))
    mat += random_noice
    return

# x [2D vector] -> feature_vector []
def feature_vector(x):
    return np.vector([1, x[0]**2, x[1]**3])

def main():
    # Generate data
    n=41; noice_var=0.3; x1 = np.linspace(-1.0, 1.0, n); x2 = np.linspace(-1.0, 1.0, n); w = np.array([0, 2.5, -0.5])
    data = generate_data(x1, x2, w, noice_var) # data = (x1,x2,t)
    # Split data & adds extra noice
    datas = split_data(data, 0.7) # datas = (test, training)
    add_noice(data_div1[0][2], 0.5**2) # Add even more noice to testdata -- e ~ N(0, 0.25**2)
    
    # Preform most liklyhood


main()

