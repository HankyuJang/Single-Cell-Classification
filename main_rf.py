import numpy as np
import sys
import argparse
from argparse import ArgumentParser
import sklearn.metrics
import classification_algo

if __name__ == '__main__':
    parser = ArgumentParser(description="gets txt file format input, then classify single cells into 9 types")
    parser.add_argument('-i', '--infile', help="Input model in npz format")
    #  parser.add_argument('-n', '--n', type=int,
            #  help="The number of trees in the forest.")
    #  parser.add_argument('-criterion', '--criterion', type=str,
            #  help="The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'entropy' for the information gain. Note: this parameter is tree-specific.")
    #  parser.add_argument('-minss', '--minss', type=int,
            #  help="The minimum number of samples required to split an internal node: If int, then consider min_samples_split as the minimum number. If float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.")
    args = parser.parse_args()

    print('-'*60)
    print("Loading the dataset...")
    # Load the dataset
    dataset = np.load(args.infile)
    trainset = dataset["Train"]
    testset = dataset["Test"]
    k_fold = dataset["Kfold"]

    # number of trees: 4, 8, ... , 4096
    # minss: 2, 4, ... , 32 
    criterion_tuple = ("gini", "entropy")
    n_list = np.power(2, np.arange(2, 13))
    minss_list = np.power(2, np.arange(1, 6))
    #####################################################################
    # Random Forest

    print('-'*60)
    print("\nRandom Forest")
    for criterion in criterion_tuple:
        for n in n_list:
            for minss in minss_list:
                accuracy_list = np.zeros(k_fold)
                for i in range(k_fold):
                    X_train = trainset[i][0]
                    y_train = trainset[i][1]
                    X_test = testset[i][0]
                    y_test = testset[i][1]
                    pred = classification_algo.random_forest(X_train, y_train, X_test, n, criterion, minss)
                    accuracy = sklearn.metrics.accuracy_score(y_test, pred)
                    #  print("Run: {}, {}".format(i+1, accuracy))
                    accuracy_list[i] = accuracy
                print("{0:.3f},criterion={1},n={2},minss={3},RandomForest".format(accuracy_list.mean(),criterion,n,minss))