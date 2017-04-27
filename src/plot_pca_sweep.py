import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse
from argparse import ArgumentParser

n = 4
SVM_l = []
kNN_l = []
NN_l = []
RF_l = []

def read_values(input_file):
    line = input_file.readline().rstrip().split(',')
    SVM_l.append(float(line[0]))
    line = input_file.readline().rstrip().split(',')
    kNN_l.append(float(line[0]))
    line = input_file.readline().rstrip().split(',')
    NN_l.append(float(line[0]))
    line = input_file.readline().rstrip().split(',')
    RF_l.append(float(line[0]))

def plot_accuracy(x, label, title, xlabel, ylabel, filename):
    plt.plot(x, SVM_l)
    plt.plot(x, kNN_l)
    plt.plot(x, NN_l)
    plt.plot(x, RF_l)
    plt.legend(label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, format=filename.split('.')[-1])
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser(description="Read the best results, then draw a line curve.")
    parser.add_argument('-i1', '--infile1', type=argparse.FileType('r'), 
            help="input file1", default=sys.stdin)
    parser.add_argument('-i2', '--infile2', type=argparse.FileType('r'), 
            help="input file2", default=sys.stdin)
    parser.add_argument('-i3', '--infile3', type=argparse.FileType('r'), 
            help="input file3", default=sys.stdin)
    parser.add_argument('-i4', '--infile4', type=argparse.FileType('r'), 
            help="input file4", default=sys.stdin)
    parser.add_argument('-i5', '--infile5', type=argparse.FileType('r'), 
            help="input file5", default=sys.stdin)
    parser.add_argument('-i6', '--infile6', type=argparse.FileType('r'), 
            help="input file6", default=sys.stdin)
    parser.add_argument('-i7', '--infile7', type=argparse.FileType('r'), 
            help="input file7", default=sys.stdin)
    parser.add_argument('-i8', '--infile8', type=argparse.FileType('r'), 
            help="input file8", default=sys.stdin)
    parser.add_argument('-i9', '--infile9', type=argparse.FileType('r'), 
            help="input file9", default=sys.stdin)
    args = parser.parse_args()


    data1 = args.infile1
    data2 = args.infile2
    data3 = args.infile3
    data4 = args.infile4
    data5 = args.infile5
    data6 = args.infile6
    data7 = args.infile7
    data8 = args.infile8
    data9 = args.infile9

    read_values(data1)
    read_values(data2)
    read_values(data3)
    read_values(data4)
    read_values(data5)
    read_values(data6)
    read_values(data7)
    read_values(data8)
    read_values(data9)

    label = ["SVM", "kNN", "NeuralNetwork", "RandomForest"]
    x = [7, 13, 25, 50, 100, 200, 400, 800, 1600]

    plot_accuracy(x, label, "Comparison of performance", "PCA num basis", "Accuracy", "plots/experiment2.png")
