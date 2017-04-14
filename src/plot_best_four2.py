import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse
from argparse import ArgumentParser

n = 8

def read_values(input_file):
    X = np.zeros(n)
    label = []
    for i in range(n):
        line = input_file.readline().rstrip().split(',')
        X[i] = float(line[0])
        label.append(line[-1])
    return X, label

def choose_best_parameters(X, label):
    X_best = np.zeros(4)
    label_best = []
    for i in range(X_best.shape[0]):
        if X[2*i] > X[2*i+1]:
            X_best[i] = X[2*i]
            label_best.append(label[2*i])
        else:
            X_best[i] = X[2*i+1]
            label_best.append(label[2*i+1])
    return X_best, label_best


def plot_accuracy(X, label, title, xlabel, ylabel, filename):
    index = np.arange(X.shape[0])
    plt.bar(index, X, align="center")
    plt.xticks(index, label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(X.min()-0.05, X.max()+0.05)
    plt.savefig(filename, format=filename.split('.')[-1])
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser(description="Read the best results, then draw a line curve.")
    parser.add_argument('-i', '--infile', type=argparse.FileType('r'), 
            help="input file", default=sys.stdin)
    parser.add_argument('-o', '--outfile', help="output file")
    parser.add_argument('-n', '--num_basis', help="number of basis")

    args = parser.parse_args()

    data = args.infile
    X, label = read_values(data)
    X, label = choose_best_parameters(X, label)

    plot_accuracy(X, label, "Comparison of performance: pca"+args.num_basis, 
                  "Classifier type", "Accuracy", args.outfile)
