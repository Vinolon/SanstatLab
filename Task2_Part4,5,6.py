from matplotlib.pylab import inv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm


def generate_data(x1, x2, w, noice_var):
    X1, X2 = np.meshgrid(x1, x2)
    random_noise = np.random.normal(0, np.sqrt(noice_var), size=(len(x2),len(x1)) )
    t = (
        np.ones((len(x2), len(x1)))*w[0] 
        + (X1**2)*w[1] 
        + (X2**3)*w[2] 
        + random_noise    
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
    num_x1 = int(len(x2)*(1-size)//2) # get between index, including index
    num_x2 = int(len(x2)*(1-size)//2)
    x1_t = x1[num_x1 : -num_x1]
    x2_t = x2[num_x2 : -num_x2]
    t_t = t[num_x2:-num_x2, num_x1:-num_x1]
    return x1_t, x2_t, t_t

def get_biased_test_data(data, size):
    x1, x2, t = data
    num_x1 = int(len(x1)*size) # get outside index, excluding index
    num_x2 = int(len(x2)*size)
    x1_t = x1[-num_x1:]
    x2_t = x2[-num_x2:]
    t_t = t[-num_x2:, -num_x1:]
    return x1_t, x2_t, t_t

def get_biased_training_data(data, size):
    x1, x2, t = data
    num_x1 = int(len(x1)*size) # get outside index, exluding index
    num_x2 = int(len(x2)*size)
    x1_t = x1[:-num_x1]
    x2_t = x2[:-num_x2]
    t_t = t[:-num_x2, :-num_x1]
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
def add_noise(mat, var):
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

def get_w_likelihood_method(alg_formatted_data):
    X, t = alg_formatted_data
    A = design_matrix(X)
    w, residuals, rank, s = np.linalg.lstsq(A, t) # max likelihood => least square method (when normal dist)
    return w

# Returns the data formatted as following
# Input:    [[x1, y2], ..., [xn, ym]]   (list of input 2-length-vectors x)
# Target:   [t11, ..., tnm]             (list of targets, each corresponding to the index of the input)
def to_algebra_approved_format(data):
    x1, x2, t = data
    X1, X2 = np.meshgrid(x1, x2)
    X = np.dstack((X1, X2))
    x_reformated = X.reshape((-1,2))  
    t_reformated = t.flatten()
    return x_reformated, t_reformated

def predict_t(x,w):
    return w[0] + w[1]*(x[0]**2) + w[2]*(x[1]**3)

# w = param, B = prediction
def predict_ts(alg_formatted_X, w):
    ones = np.ones(len(alg_formatted_X))
    X1 = alg_formatted_X[:,0]
    X2 = alg_formatted_X[:,1]
    t_pred = ones*w[0] + (X1**2)*w[1] + (X2**3)*w[2]
    return t_pred

def predict_B(data, w):
    X, t_real = data
    t_pred = predict_ts(X, w)
    var_pred = np.mean((t_real-t_pred)**2)
    return 1/var_pred

def mean_squared_error(t_true, t_pred):
    return np.mean((t_true - t_pred) ** 2)

def Bayesian_variance(alpha, b, training_data):
    X_train, t = training_data
    
    featureX = design_matrix(X_train)
    featureX_T = featureX.T
    B = b
    a = alpha
    I = np.eye(featureX.shape[1])
    
    sNinv = a*I + B*(featureX_T@featureX)
    sN = inv(sNinv)
    mN = B*sN@(featureX_T@t)
    
    return sN, mN

def Bayesian_pred(sN, mN, test_data, b):
    X_test, t = test_data
    
    pred_variance = []
    pred_mean = []
    featureX = design_matrix(X_test)
    
    for x in featureX:
        variance = 1/b + (x.T)@sN@x
        mean = mN.T@x
        
        
        pred_variance.append(variance)
        pred_mean.append(mean)
    
    return pred_variance, pred_mean
    

def main():
    # Generate data
    n=41; sigma2=0.3; 
    x1 = np.linspace(-1.0, 1.0, n); x2 = np.linspace(-1.0, 1.0, n); 
    w = np.array([0, 2.5, -0.5]); b = 1/sigma2
    data = generate_data(x1, x2, w, sigma2)
    datas = split_data(data, 0.7) # datas = (test, training)
    add_noise(datas[0][2], 0.25**2) # Add even more noise to testdata -- e ~ N(0, 0.25**2) (Could it be N(0, 0.25**2*sigma2)?)
    train_data_alg_format = to_algebra_approved_format(datas[1])
    test_data_alg_format = to_algebra_approved_format(datas[0])
    
    w_ML = get_w_likelihood_method(train_data_alg_format)
    t_prediction_ML = predict_ts(test_data_alg_format[0], w_ML)
    mse_ML = mean_squared_error(test_data_alg_format[1], t_prediction_ML)
    
    for alpha in [0.2, 0.8, 2.0]:
        
        sN, mN = Bayesian_variance(alpha, b, train_data_alg_format)
        pred_variance_test, pred_mean_test = Bayesian_pred(sN, mN, test_data_alg_format, b)
        t_pred_test = pred_mean_test
        mean_variance_test = np.mean(pred_variance_test)
        mse_Bay = mean_squared_error(test_data_alg_format[1], t_pred_test)
        
        pred_variance_train, pred_mean_train = Bayesian_pred(sN, mN, train_data_alg_format, b)
        t_pred_train = pred_mean_train
        mean_variance_train = np.mean(pred_variance_train)
        mse_Bay_train = mean_squared_error(train_data_alg_format[1], t_pred_train)
    
        print(f"Mean of test variance for alpha {alpha}: {mean_variance_test}")
        print(f"MSE of Bayesian approach for test data: \n\t {mse_Bay}")
        print(f"MSE of ML method:\n\t {mse_ML}")
        
        print(f"Mean of train variance for alpha {alpha}: {mean_variance_train}")
        print(f"MSE of Bayesian approach for train data: \n\t {mse_Bay_train}")


main()

#Task 2 part 6: It's clear that getting the exact same data to predict will ensure the Bayesian approach is extremely accurate.
#The mean of its variance is essentially the same as the introduced noise (sigma2+0.25**2). The greater the alpha the better the prediction 
# becomes for the mean, but the mean of the variance decreases when the alpha increases.
