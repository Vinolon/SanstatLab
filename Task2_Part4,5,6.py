from matplotlib.pylab import inv
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import multivariate_normal
#from scipy.stats import norm

def generate_data(x1, x2, w, noice_var):
    X1, X2 = np.meshgrid(x1, x2)
    random_noice = np.random.normal(0, np.sqrt(noice_var), size=(len(x2),len(x1)) )
    t = (
        np.ones((len(x2), len(x1)))*w[0] 
        + (X1**2)*w[1] 
        + (X2**3)*w[2] 
        + random_noice    
    )
    
    return to_algebra_aproved_format((x1, x2, t))

# Splits into test and train data
def split(data, cutof):
    x, t = data
    x_test = []
    t_test = []
    x_train = []
    t_train = []
    for i in range(len(x)):
        if abs(x[i,0]) >= cutof or abs(x[i,1]) >= cutof:
            x_test.append(x[i])
            t_test.append(t[i])
        else:
            x_train.append(x[i])
            t_train.append(t[i])
    x_test = np.array(x_test)
    t_test = np.array(t_test)
    x_train = np.array(x_train)
    t_train = np.array(t_train)
    return ((x_test, t_test), (x_train, t_train))

def split_biased(data, cutof):
    x, t = data
    x_test = []
    t_test = []
    x_train = []
    t_train = []
    for i in range(len(x)):
        if x[i,0]>=cutof or x[i,1]>=cutof:
            x_test.append(x[i])
            t_test.append(t[i])
        else:
            x_train.append(x[i])
            t_train.append(t[i])
    x_test = np.array(x_test)
    t_test = np.array(t_test)
    x_train = np.array(x_train)
    t_train = np.array(t_train)
    return ((x_test, t_test), (x_train, t_train))
    
# Adds noice ~ N(0, var) to a matrix mat
# Mutates mat, Returns nothing
def add_noice(mat, var, magnitute):
    random_noice = magnitute*np.random.normal(0, np.sqrt(var), size=(mat.shape))
    mat += random_noice
    return

# X [list of D-dim-vectors (x)]
# Returns design matrix
def design_matrix(X):
    # x [2D vector] -> feature_vector []
    def feature_vector(x):
        return np.array([1, x[0]**2, x[1]**3])
    return np.array(list(map(feature_vector, X)))

def get_w_likelihood_method(alg_formated_data):
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
    n=41; normvar = 0.3; normsize= 0.5

    x1 = np.linspace(-1.0, 1.0, n); x2 = np.linspace(-1.0, 1.0, n); 
    w = np.array([0, 2.5, -0.5]); 
    data = generate_data(x1, x2, w, normvar)
    datas = split_biased(data, normsize) # datas = (test, training)
    add_noice(datas[0][1], normvar, 0.25) # Add even more noise to testdata -- 
    b = 1/normvar
    normproportion_train_biased = ((1+normsize)**2)/4


    #alpha = 0.8
    mse_ML_arr = []
    mse_Bay_arr = []
    mse_Bay_train_arr = []
    variance = [0.1,0.2,0.3]
    train_size = [0.3, 0.4, 0.5, 0.6]
    proportion_arr = []
    normalpha = [0.2, 0.8, 2.0]
    
    for alpha in [0.2, 0.8, 2.0]:
        #data = generate_data(x1, x2, w, normvar)
        #datas = split_biased(data, size) # datas = (test, training)
        #add_noice(datas[0][1], normvar, 0.25) # Add even more noise to testdata -- 
        train = datas[1]
        test = datas[0]
        #b = 1/var
        #proportion_train = ((1+0.3)**2)/4

        w_ML = get_w_likelihood_method(train)
        t_prediction_ML = predict_ts(test[0], w_ML)
        mse_ML = mean_squared_error(test[1], t_prediction_ML)

        sN, mN = Bayesian_variance(alpha, b, train)
        pred_variance_test, pred_mean_test = Bayesian_pred(sN, mN, test, b)
        t_pred_test = pred_mean_test
        mean_variance_test = np.mean(pred_variance_test)
        mse_Bay = mean_squared_error(test[1], t_pred_test)
        
        pred_variance_train, pred_mean_train = Bayesian_pred(sN, mN, train, b)
        t_pred_train = pred_mean_train
        mean_variance_train = np.mean(pred_variance_train)
        mse_Bay_train = mean_squared_error(train[1], t_pred_train)

        mse_Bay_arr.append(mse_Bay)
        mse_Bay_train_arr.append(mse_Bay_train)
        mse_ML_arr.append(mse_ML)
        #proportion_arr.append(proportion_train)
    
        print(f"Mean of test variance for alpha {alpha}: {mean_variance_test}")
        print(f"MSE of Bayesian approach for test data: \n\t {mse_Bay}")
        print(f"MSE of ML method:\n\t {mse_ML}")
        
        print(f"Mean of train variance for alpha {alpha}: {mean_variance_train}")
        print(f"MSE of Bayesian approach for train data: \n\t {mse_Bay_train}")

        X_test = datas[0][0]
        t_test = datas[0][1]
        X_train = datas[1][0]
        t_train = datas[1][1]
        X = data[0]
        """
        fig1 = plt.figure()
        ax = fig1.add_subplot(111, projection='3d')
        ax.set_title(f"Mean of test data variance: {round(mean_variance_test, 4)}\n \n alpha: {alpha}")

        ax.scatter(X_test[:, 0], X_test[:, 1], t_test, color = "blue")
        ax.scatter(X_test[:, 0], X_test[:, 1], t_pred_test, color = "red")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("t")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Mean of training data variance: {round(mean_variance_train, 4)}\n alpha: {alpha}")

        ax.scatter(X_train[:, 0], X_train[:, 1], t_train, color = "blue")
        ax.scatter(X_train[:, 0], X_train[:, 1], t_pred_train, color = "red")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("t")
        plt.legend()
        plt.tight_layout()
        plt.show()"""

    #plt.title(f"{round(normproportion_train_biased, 4)}% amount of biased training data")
    plt.title(f"Test data mean variance vs training data mean variance, \n {round(normproportion_train_biased, 4)}% amount of biased training data")

    plt.plot(normalpha, mse_Bay_train_arr, color = "red", label = "Training data mean variance")
    plt.plot(normalpha, mse_Bay_arr, color = "blue", label = "Testing data mean variance")
    plt.xlabel("alpha")
    plt.ylabel("mean")
    plt.legend()
    plt.tight_layout()
    plt.show()


main()

#Task 2 part 6: It's clear that getting the exact same data to predict will ensure the Bayesian approach is extremely accurate.
#The mean of its variance is essentially the same as the introduced noise (sigma2+0.25**2). The greater the alpha the better the prediction 
# becomes for the mean, but the mean of the variance decreases when the alpha increases.
