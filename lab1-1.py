import numpy as np
import matplotlib
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

mem = Memory("./lab1cache")

@mem.cache
def get_data():
    data = load_svmlight_file("housing_scale")
    return data[0], data[1]


def f(x, w, b):
    y = b
    for index in range(len(x)):
        y += w[index]*x[index]
    return y


def gradient(x, xt, y, w):
    result = xt * x * w - xt*y
    return result


def loss_function(x, y, w):
    result = 0.5 * (y.T*y - x * w).T*(y - x * w)
    return result

if __name__ == "__main__":

    # 1,import data
    X, Y = get_data()

    # 2,split data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

    # CSR_matrix to array
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    # add 1 for Xi
    array_with_1 = [[1]]*len(x_train)
    x_train = np.column_stack((x_train, array_with_1))
    array_with_1 = [[1]] * len(x_test)
    x_test = np.column_stack((x_test, array_with_1))

    # 3,initialize data
    w = [[0]]*len(x_train[0])
    learning_rate = 0.0005
    rounds = 10000
    l_train_array = [None] * rounds
    l_validation = [None] * rounds

    # pre process
    x_train = np.mat(x_train)
    y_train = np.mat(y_train)
    y_train = y_train.T
    x_T = x_train.T

    x_test = np.mat(x_test)
    y_test = np.mat(y_test)
    y_test = y_test.T

    for i in range(rounds):

        # 5, get the Gradient
        g = gradient(x_train, x_T, y_train, w)

        # 6,7 update the w
        w = w - g * learning_rate

        # 8,loss function value
        loss_train = loss_function(x_train, y_train, w)
        loss_train = loss_train[0, 0]
        print(str(i) + "loss in train: " + str(loss_train))
        l_train_array[i] = loss_train

        loss_validation = loss_function(x_test, y_test, w)
        loss_validation = loss_validation[0, 0]
        print(str(i) + "loss in validation: " + str(loss_validation))
        l_validation[i] = loss_validation

    plt.plot(range(len(l_train_array)), l_train_array, label="train")
    plt.plot(range(len(l_validation)), l_validation, label="validation")
    plt.grid(True)
    plt.legend()
    plt.show()

