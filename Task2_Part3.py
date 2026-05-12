import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm

# Functions ########################################################

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

def predict_B(data, w):
    X, t_real = data
    t_pred = predict_ts(X, w)
    var_pred = np.mean((t_real-t_pred)**2)
    return 1/var_pred

def mean_squared_error(t_true, t_pred):
    return np.mean((t_true - t_pred) ** 2)

# Sub Tasks ########################################################
def task2_part1():
    # - Generate data -
    # Generate and plot data points using the model defined in Eq. 38 over the 2D domain of
    # the input space where x = [x1, x2] subset of [−1, −0.95, . . . , 0.95, 1] x [−1, −0.95, . . . , 0.95, 1].
    # Please use some fixed value of data noise σ, say, σ2 = 0.3 (try also σ2 subset of {0.6, 1.2})
    
    n=41; x1 = np.linspace(-1.0, 1.0, n); x2 = np.linspace(-1.0, 1.0, n); w = np.array([0, 2.5, -0.5])

    fig = plt.figure()
    for i in range(4):
        # Generate data
        noice_var = [0.0,0.3,0.6,1.2][i]
        data = generate_data(x1, x2, w, noice_var) # data = (x1,x2,t)

        # Plots
        X = data[0]
        t = data[1]

        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], t, c=t)

        ax.set_title(f"noise = {noice_var}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("t")
    plt.tight_layout()
    plt.show()
    
    return

def task2_part2():
    #Choose the samples for which |x1| > 0.3 or |x2| > 0.3 as your test data and keep the
    #rest as your training subset (please consider also alternative scenarios where the training
    #data are downsampled down to 5-10% of the original training dataset size and are biased,
    #say, x1>0.3 and x2>0.3). Next, to simulate more reaalistic conditions about testing
    #uncertainties, please add extra zero-mean noise to the test data (t SIextra; SIextra ' 0.25*SI).
    n=41; noice_var=0.3; x1 = np.linspace(-1.0, 1.0, n); x2 = np.linspace(-1.0, 1.0, n); w = np.array([0, 2.5, -0.5])
    print(f"Origional params w:\n\t {w}")
    data = generate_data(x1, x2, w, noice_var) # data = (x1,x2,t)

    # SPLITS DATA
    # yields all |x1|, |x2| > 0.3
    unbiased_split_data = split(data, 0.3) # datas = (test, training)
    # yields all x1, x2 > 0.3
    biased_split_data = split_biased(data, 0.3) # datas = (test, training)

    # ADDS EXTRA NOICE to test datasets
    add_noice(unbiased_split_data[0][1], 0.3, 0.25)
    add_noice(biased_split_data[0][1], 0.3, 0.25)

    fig = plt.figure()
    for i in range(2):
        # Plots
        header = ["Unbiased choice of Training & Testing data", "Biased choice of Training & Testing data"][i]
        datas = [unbiased_split_data, biased_split_data][i]

        X_test = datas[0][0]
        t_test = datas[0][1]
        X_train = datas[1][0]
        t_train = datas[1][1]

        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.scatter(X_test[:, 0], X_test[:, 1], t_test)
        ax.scatter(X_train[:, 0], X_train[:, 1], t_train)

        ax.set_title(header)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("t")
    plt.tight_layout()
    plt.show()
    
    return


def task2_part3():
    # Generate data
    n=41; noice_var=0.3; x1 = np.linspace(-1.0, 1.0, n); x2 = np.linspace(-1.0, 1.0, n); w = np.array([0, 2.5, -0.5]); is_biased=False; split_cutof=0.3

    print(f"Origional params w:\n\t {w}")
    #n=11; noice_var=0; x1 = np.linspace(0.0, 10.0, n); x2 = np.linspace(0.0, 10.0, n); w = np.array([0, 2, -1]) # TEST VALUES
    data = generate_data(x1, x2, w, noice_var) # data = (x1,x2,t)
    # Split data & adds extra noice
    if not is_biased:
        datas = split(data, split_cutof) # datas = (test, training)
    else:
        datas = split_biased(data, split_cutof) # datas = (test, training)
    add_noice(datas[0][1], noice_var, 0.25) # Add even more noice to testdata --  ##TOLKNING AV INSTRUNKTIONERNA STEG 2- ÄR DETTA RÄTT????
    # Preform most liklyhood
    w_ML = get_w_liklyhood_method(datas[1])
    print(f"Predicted params w with Most-Liklyhood method:\n\t {w_ML}")
    B_ML = predict_B(datas[1], w_ML)
    print(f"Predicted precition B and variance V with Most-Liklyhood method:\n\t B:{B_ML}, V:{1/B_ML}")
    # Test ML with MSE
    t_prediction_ML = predict_ts(datas[0][0], w_ML)
    mse_ML = mean_squared_error(datas[0][1], t_prediction_ML)
    print(f"MSE of ML method:\n\t {mse_ML}")

    #PLOTS
    fig = plt.figure()

    X_test = datas[0][0]
    t_test = datas[0][1]
    X_train = datas[1][0]
    t_train = datas[1][1]
    X = data[0]

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(X_test[:, 0], X_test[:, 1], t_test)
    ax.scatter(X_train[:, 0], X_train[:, 1], t_train)
    ax.scatter(X_test[:, 0], X_test[:, 1], t_prediction_ML)

    if not is_biased:
        propotion_tain = ((split_cutof*2)**2)/4
    else:
        propotion_tain = ((1+split_cutof)**2)/4

    ax.set_title(f"Liklyhood method\nNoiceVar: {noice_var}\nBiased split: {is_biased}\nTrain data: {propotion_tain*100}%\nMSE: {mse_ML}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("t")
    plt.tight_layout()
    plt.show()


def plotML():

    return

def main():
    #task2_part1()
    #task2_part2()
    for i in range(5):
        task2_part3()
    return


main()
    
