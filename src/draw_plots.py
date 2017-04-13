import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse
from argparse import ArgumentParser

GRID_WIDTH_KNN = 8
GRID_HEIGHT_KNN = 2
GRID_WIDTH_RF = 8
GRID_HEIGHT_RF = 2

def plot_heatmap_knn(data, title, xlabel, ylabel, filename):
    plt.matshow(data, extent=[3, 18, 0, 2], aspect='auto',
                cmap="plasma", vmin=0, vmax=1)
    plt.colorbar()
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.text(-0.25, 2.25, 'S', fontsize=35)
    # plt.text(7.75, 0.25, 'G', fontsize=35)
    plt.savefig(filename, format=filename.split('.')[-1])

def read_values_rf(input_file):
    # skip 4 lines
    for i in range(5):
        input_file.readline()

    X = np.zeros((GRID_HEIGHT_RF, GRID_WIDTH_RF))
    for i in range(GRID_HEIGHT_RF):
        for j in range(GRID_WIDTH_RF):
            line = input_file.readline().rstrip().split(',')
            accuracy = line[0]
            X[i][j] = accuracy
    return X

def read_values_knn(input_file):
    # skip 4 lines
    for i in range(4):
        input_file.readline()

    X = np.zeros((GRID_HEIGHT_KNN, GRID_WIDTH_KNN))
    for i in range(GRID_HEIGHT_KNN):
        for j in range(GRID_WIDTH_KNN):
            line = input_file.readline().rstrip().split(',')
            accuracy = line[0]
            X[i][j] = accuracy
    return X

if __name__ == '__main__':
    parser = ArgumentParser(description="Value heatmap")
    parser.add_argument('-knn', '--knn', type=argparse.FileType('r'), 
            help="value", default=sys.stdin)
    args = parser.parse_args()

    v = args.knn
    X_v = read_values_knn(v)

    plot_heatmap_knn(X_v, "KNN accuracy",
            "number of k (nearest neighbors)", "", "knn_uniform.png")
