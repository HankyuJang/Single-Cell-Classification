import numpy as np
import sys
import argparse
from argparse import ArgumentParser
import classification_algo
import sklearn.metrics

if __name__ == '__main__':
    parser = ArgumentParser(description="gets txt file format input, then classify single cells into 9 types")
    parser.add_argument('-i', '--infile', help="Input model in npz format")
    parser.add_argument('-hls', '--hls', nargs='+', type=int,
             help="The ith element represents the number of neurons in the ith hidden layer.")
    parser.add_argument('-activation', '--activation', type=str,
             help="Activation function for the hidden layer. 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)). 'tanh', the hyperbolic tan function, returns f(x) = tanh(x). 'relu', the rectified linear unit function, returns f(x) = max(0, x)")
    parser.add_argument('-solver', '--solver', type=str,
             help="The solver for weight optimization. 'lbfgs' is an optimizer in the family of quasi-Newton methods. 'sgd' refers to stochastic gradient descent. 'adam' refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba Note: The default solver 'adam' works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, 'lbfgs' can converge faster and perform better.")
    parser.add_argument('-alpha', '--alpha', type=float,
             help="L2 penalty (regularization term) parameter.")
    args = parser.parse_args()

    if args.hls!=None:
         hls = tuple(args.hls)

    # Load the dataset
    dataset = np.load(args.infile)
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    k_fold = dataset["Kfold"]

    alpha_list = np.power(2.0, np.arange(-14, 1))
    alpha = args.alpha
    activation = args.activation
    solver = args.solver

    #####################################################################
    # Neural Network
    for i in range(k_fold):
        pred = classification_algo.neural_network(X_train[i], y_train[i], X_test[i], hls, args.activation, args.solver, alpha)
        accuracy = sklearn.metrics.accuracy_score(y_test[i], pred)
        print("{0:.3f},hls={1},alpha={2},activation={3},solver={4},NeuralNetwork".format(accuracy,hls,alpha,args.activation,args.solver))
