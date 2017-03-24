#!/bin/bash

python main_rf.py -i dataset.npz > rf.dat
python main_svm.py -i dataset.npz > svm.dat
python main_knn.py -i dataset.npz > knn.dat
python main_nn.py -i dataset.npz > nn.dat

