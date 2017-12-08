import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random

mem = Memory("./lab2cache")


@mem.cache
def get_data():
    data = load_svmlight_file("australian_scale")
    return data[0], data[1]


def gw(x, y, w, b):
    _ = y[0, 0] * ( (w.T * x.T)[0, 0] + b)
    if 1 - _ >= 0:
        return - y[0, 0] * x
    else:
        return 0


def gradient_w(y, w, x, b, c):
    result = w
    for i in range(len(x)):
        gwi = c * gw(x[i], y[i], w, b)

        # gw is not 0
        if type(gwi) is np.matrix:
            result += gwi.T
    return result


def gb(x, y, w, b):
    _ = y[0, 0] * ((w.T * x.T)[0, 0] + b)
    if 1 - _ >= 0:
        y = -y[0, 0]
        return y
    else:
        return 0


def gradient_b(y, w, x, b, c):
    result = 0
    for i in range(len(x)):
        result += c * gb(x[i], y[i], w, b)
    return result


def loss_function(x, y, w, b, c):
    result = 0
    for i in w:
        result += i[0, 0] * i[0, 0]
    for i in range(len(x)):
        _ = c * max(0, 1 - y[i, 0] * ((w.T * x[i].T)[0, 0] + b ))
        result += _
    return result


def random_select(x, y, size=50):
    _ = random.random()
    index = int(_ * len(x))
    end = index + size
    if size < len(x):
        return x[index:end, :], y[index:end, :]
    else:
        return x[index:, :], y[index:, :]

if __name__ == "__main__":
    # 1,import data
    X, Y = get_data()

    # 2,split data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    # CSR_matrix to array
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    # 3,initialize data
    w = [[0]] * len(x_train[0])
    w = np.mat(w, dtype=np.float64)
    learning_rate = 0.05
    rounds = 6000
    batch_size = 50
    C = 10
    b = 0
    success_rate_test = [None] * rounds
    success_rate_train = [None] * rounds
    loss_list = [None] * rounds

    # pre process
    x_train = np.mat(x_train)
    y_train = np.mat(y_train)
    y_train = y_train.T

    x_test = np.mat(x_test)
    y_test = np.mat(y_test)
    y_test = y_test.T

    # Gradient Descent
    for i in range(rounds):
        w = w - gradient_w(y_train, w, x_train, b, C) * learning_rate
        b = b - gradient_b(y_train, w, x_train, b, C) * learning_rate

        # count success rate in training set:
        success_train = 0
        x_train_batch, y_train_batch = random_select(x_train, y_train, size=batch_size)
        for i2 in range(len(y_train_batch)):
            # print(x_test[i2, :])
            if x_train_batch[i2, :] * w + b >= 0:
                _ = -1
            else:
                _ = 1
            if _ == y_train_batch[i2, 0]:
                success_train += 1
        success_rate_train[i] = success_train/len(y_train_batch)

        # count success rate in testing set:
        success_test = 0
        x_test_batch, y_test_batch = random_select(x_test, y_test,size=batch_size)
        for i2 in range(len(y_test_batch)):
            # print(x_test[i2, :])
            if x_test_batch[i2, :] * w + b >= 0:
                _ = -1
            else:
                _ = 1
            if _ == y_test_batch[i2, 0]:
                success_test += 1
        success_rate_test[i] = success_test / len(y_test_batch)

        # count loss
        loss = loss_function(x_train, y_train, w, b, C)
        loss_list[i] = loss

        # print("i = " + str(i) + "    test success rate:" + str(success_test / len(y_test)) + "loss: " + str(loss))
        print("i = " + str(i) + "    train success rate:" + str(success_train / batch_size) +
              "    test success rate:" + str(success_test / batch_size) + "loss: " + str(loss))

    # plt.plot(range(rounds), success_rate_train, label="train")
    # plt.plot(range(rounds), success_rate_test, label="validation")
    plt.plot(range(rounds), loss_list)
    plt.show()
