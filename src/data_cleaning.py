import numpy as np
import sys
import argparse
from argparse import ArgumentParser
import sklearn.model_selection
import sklearn.metrics

def get_X_y(infile):
    # Skip irrelevant lines
    for i in range(7):
        infile.readline()
    header = infile.readline()
    header = header.split()

    for i in range(3):
        infile.readline()

    X = np.zeros((4998, 3005)).astype(int)
    y = np.zeros(4998).astype(int)

    for i in range(4998):
        line = infile.readline()
        line = line.split()
        y[i] = int(line[1])
        X[i] = [int(value) for value in line[2:]]

    for line in infile:
        continue

    return X, y

def prepare_train_test_using_kfold(k, X, y):
    trainset, testset = [], []
    kf = sklearn.model_selection.KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(X)
    print(kf)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        trainset.append((X_train, y_train))
        testset.append((X_test, y_test))
    return trainset, testset

if __name__ == '__main__':
    parser = ArgumentParser(description="gets txt file format input, then classify single cells into 9 types")
    parser.add_argument('-i', '--infile', type=argparse.FileType('r'), 
            help="txt file format (line 8: header, line 12 ~ line5009: data - col1: name - col2: label col3 ~ col: activation levels)", default=sys.stdin)
    parser.add_argument('-o', '--outfile', help="output file name")
    parser.add_argument('-kfold', '--kfold', type=int, default=5, help="k-fold (positive integer. default is 5-fold)")
    args = parser.parse_args()

    k_fold = args.kfold
    #####################################################################
    # Prepare X and y from the input txt file
    print("Preparing X and y from the input txt file...")
    X, y = get_X_y(args.infile)

    #####################################################################
    # Get trainsets and testsets using K-Fold
    print("Preparing the trainsets and testsets using {} fold...".format(k_fold))
    trainset, testset = prepare_train_test_using_kfold(k_fold, X, y)

    np.savez(args.outfile, Train = trainset, Test = testset, Kfold = k_fold)
    print("-"*60)
    print("trainset and testset are created!")
    print("They are saved as `dataset.npz` in the same directory")
