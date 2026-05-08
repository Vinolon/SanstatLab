import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm


def generate_data(x1, x2, w, noice_var):
    X1, X2 = np.meshgrid(x1, x2)
    random_noice = np.random.normal(0, np.sqrt(noice_var), size=(len(x2),len(x1)) )
    t = (
        np.ones((len(x2), len(x1)))*w[0] 
        + (X1**2)*w[1] 
        + (X2**3)*w[2] 
        + random_noice    
    )
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
    test_data = get_test_data(data, test_size)
    train_data = get_training_data(data, 1-test_size)
    return test_data, train_data

# Splits the data into <test_size>% test data and the rest trainging data
# Uses a biased splitting aproach
def split_data_biased(data, test_size):
    test_data = get_biased_test_data(data, test_size)
    train_data = get_biased_training_data(data, 1-test_size)
    return test_data, train_data
    
# Adds noice ~ N(0, var) to a matrix mat
# Mutates mat, Returns nothing
def add_noice(mat, var):
    random_noice = np.random.normal(0, np.sqrt(var), size=(mat.shape))
    mat += random_noice
    return

# X [list of D-dim-vectors (x)]
# Returns design matrix
def design_matrix(X):
    # x [2D vector] -> feature_vector []
    def feature_vector(x):
        return np.array([1, x[0]**2, x[1]**3])
    return np.array(list(map(feature_vector, X)))

def get_w_liklyhood_method(alg_formated_data):
    X, t = alg_formated_data
    A = design_matrix(X)
    w, residuals, rank, s = np.linalg.lstsq(A, t) # max liklyhood => least square method (when normal dist)
    return w

# Returns the data formatted as following
# Input:    [[x1, y2], ..., [xn, ym]]   (list of input 2-length-vetors x)
# Target:   [t11, ..., tnm]             (list of targets, each coresponding to the index of the input)
def to_algebra_aproved_format(data):
    x1, x2, t = data
    X1, X2 = np.meshgrid(x1, x2)
    X = np.dstack((X1, X2))
    x_reformated = X.reshape((-1,2))  
    t_reformated = t.flatten()
    return x_reformated, t_reformated

def predict_t(x,w):
    return w[0] + w[1]*(x[0]**2) + w[2]*(x[1]**3)

# w = param, B = preition
def predict_ts(alg_formated_X, w):
    ones = np.ones(len(alg_formated_X))
    X1 = alg_formated_X[:,0]
    X2 = alg_formated_X[:,1]
    t_pred = ones*w[0] + (X1**2)*w[1] + (X2**3)*w[2]
    return t_pred

def predict_B(t_real, t_pred):
    var_pred = np.mean((t_real-t_pred)**2)
    return 1/var_pred

def mean_squared_error(t_true, t_pred):
    return np.mean((t_true - t_pred) ** 2)

def main():
    # Generate data
    n=41; noice_var=0.3; x1 = np.linspace(-1.0, 1.0, n); x2 = np.linspace(-1.0, 1.0, n); w = np.array([0, 2.5, -0.5])
    print(f"Origional params w:\n\t {w}")
    #n=11; noice_var=0; x1 = np.linspace(0.0, 10.0, n); x2 = np.linspace(0.0, 10.0, n); w = np.array([0, 2, -1]) # ETST VALUES
    data = generate_data(x1, x2, w, noice_var) # data = (x1,x2,t)
    # Split data & adds extra noice
    datas = split_data(data, 0.7) # datas = (test, training)
    add_noice(datas[0][2], 0.25**2) # Add even more noice to testdata -- e ~ N(0, 0.25**2)  ##TOLKNING AV INSTRUNKTIONERNA STEG 2- ÄR DETTA RÄTT????
    # Preform most liklyhood
    train_data_alg_format = to_algebra_aproved_format(datas[1])
    w_ML = get_w_liklyhood_method(train_data_alg_format)
    print(f"Predicted params w with Most-Liklyhood method:\n\t {w_ML}")
    test_data_alg_format = to_algebra_aproved_format(datas[0])
    t_prediction_ML = predict_ts(test_data_alg_format[0], w_ML)
    B_ML = predict_B(train_data_alg_format[1], t_prediction_ML)
    print(f"Predicted precition B with Most-Liklyhood method:\n\t {B_ML}")
    # Test ML with MSE
    mse_ML = mean_squared_error(test_data_alg_format[1], t_prediction_ML)
    print(f"MSE of ML method:\n\t {mse_ML}")


main()

