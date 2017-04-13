import numpy as np
import sys
import argparse
from argparse import ArgumentParser
import classification_algo
import sklearn.metrics

if __name__ == '__main__':
    parser = ArgumentParser(description="gets txt file format input, then classify single cells into 9 types")
    parser.add_argument('-i', '--infile', help="Input model in npz format")
    #  parser.add_argument('-n', '--n', type=int,
            #  help="Number of neighbors to use by default for k_neighbors queries.")
    #  parser.add_argument('-weights', '--weights', type=str,
            #  help="weight function used in prediction. Possible values: 'uniform' : uniform weights. All points in each neighborhood are weighted equally. 'distance' : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.")
    args = parser.parse_args()

    #  print("\nAfter run, the program will print accuracy of the train and test set")

    print('-'*60)
    print("Loading the dataset...")
    # Load the dataset
    dataset = np.load(args.infile)
    trainset = dataset["Train"]
    testset = dataset["Test"]
    k_fold = dataset["Kfold"]

    max_n = 19
    weights_tuple = ("uniform", "distance")

    print('-'*60)
    print("k nearest algorithm")
    
    for weights in weights_tuple:
        for n in range(3, max_n, 2):
            accuracy_list = np.zeros(k_fold)
            for i in range(k_fold):
                X_train = trainset[i][0]
                y_train = trainset[i][1]
                X_test = testset[i][0]
                y_test = testset[i][1]
                pred = classification_algo.k_nearest_neighbor(X_train, y_train, X_test, n, weights)
                accuracy = sklearn.metrics.accuracy_score(y_test, pred)
                #  print("Run: {}, {}".format(i+1, accuracy))
                accuracy_list[i] = accuracy
            #  print("\nAverage accuracy for kNN classifier: {0:.3f}".format(accuracy.mean()))
            print("{0:.3f},n={1},weights={2},kNN".format(accuracy_list.mean(),n,weights))
