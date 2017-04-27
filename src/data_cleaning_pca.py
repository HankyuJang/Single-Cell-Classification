import numpy as np
import sys
import argparse
from argparse import ArgumentParser
import sklearn.model_selection
import sklearn.metrics
from sklearn.decomposition import PCA

def get_X_y(infile):
    # Skip 'tissue' (not using in the classification)
    infile.readline()
    # Use group labels here for the labels
    group = infile.readline().split()
    y = np.array(group[2:]).astype(int)

    # Skip lines not used in the classification
    for i in range(5):
        infile.readline()

    # Get cell ids
    cell_id = infile.readline().split()
    header = np.array(cell_id[2:])

    # Skip lines not used in the classification
    for i in range(3):
        infile.readline()

    X = np.zeros((4998, 3005)).astype(int)

    # Save the data in X matrix (each column is the vector of one cell with gene expressions)
    for i in range(4998):
        line = infile.readline().split()
        X[i] = line[2:]

    # Read the rest of files
    for line in infile:
        continue

    return X.T, y

def prepare_train_test_using_kfold(k, X, y):
    trainset, testset = [], []
    kf = sklearn.model_selection.KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(X)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    print(kf)
    for train_index, test_index in kf.split(X):
        X_train.append(X[train_index])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    parser = ArgumentParser(description="gets txt file format input, then classify single cells into 9 types")
    parser.add_argument('-i', '--infile', type=argparse.FileType('r'), 
            help="Use expressionmRNAAnnotations.txt", default=sys.stdin)
    parser.add_argument('-o', '--outfile', help="output file name")
    parser.add_argument('-kfold', '--kfold', type=int, default=5, help="k-fold (positive integer. default is 5-fold)")
    parser.add_argument('-n', '--num_basis', type=int, default=100, help="number of basis vectors to use for dimension reduction")
    args = parser.parse_args()

    k_fold = args.kfold
    n = args.num_basis
    #####################################################################
    # Prepare X and y from the input txt file
    print("Preparing X and y from the input txt file...")
    X, y = get_X_y(args.infile)
    
    #####################################################################
    # project the feature space up n dimensions
    pca = PCA(n_components=n)
    pca.fit(X)
    X = pca.transform(X)

    #####################################################################
    # Get trainsets and testsets using K-Fold
    print("Preparing the trainsets and testsets using {} fold...".format(k_fold))
    X_train, y_train, X_test, y_test = prepare_train_test_using_kfold(k_fold, X, y)

    np.savez(args.outfile, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, Kfold = k_fold)
    print("-"*60)
    print("trainset and testset are created!")
