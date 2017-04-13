import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse
from argparse import ArgumentParser

n = 4

def read_values(input_file):
    X = np.zeros(n)
    label = []
    for i in range(n):
        line = input_file.readline().rstrip().split(',')
        X[i] = float(line[0])
        label.append(line[-1])
    return X, label

def plot_accuracy(X, label, title, xlabel, ylabel, filename):
    index = np.arange(X.shape[0])
    plt.bar(index, X)
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
    args = parser.parse_args()

    data = args.infile
    X, label = read_values(data)
    plot_accuracy(X, label, "Comparison of performance", "Classifier type", "Accuracy", "plots/experiment1.png")
