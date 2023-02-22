import numpy as np
from Layer.py import Layer
from Model.py import Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets,preprocessing
data_set = datasets.load_breast_cancer()
x_train,x_test = data_set['data'][:430].T,data_set['data'][430:].T
y_train,y_test = data_set['target'][:430].reshape(1,-1),data_set['target'][430:].reshape(1,-1)


def sigmoid(Z):
    return 1 / (1 + np.exp(-1*Z))


def relu(x):
    np.putmask(x,x<0,0)
    return x


def relu_derivative(x):
    np.putmask(x,x>0,1)
    return x


def main():
    model = Model(Layer(8),Layer(16),Layer(16),Layer(64),Layer(1),X = x_train, Y=y_train)
    model.fit()
    model.train(50000)
    accuracy = model.test(x_test,y_test)

if __name__ == '__main__':
    main()
