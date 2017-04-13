# Classifying Single-Cell Types from Mouse Brain RNA-Seq Data using Machine Learning Algorithms

-----

## Team mebers

- Samer Al-Saffar
- Hankyu Jang

## Datasets

- line 8: Header
- line 12 ~ line 5009: data with labels from 1 to 9
- data: column 1: cell name, column 2: label, column 3: activation level

### Preprocessing

`data_cleaning.py`

- Input
    - expressionmRNAAnnotations.txt
    - k (integer)
- Procedure
    - Cleans the data, then creates data matrix X and corresponding label vector y
    - Prepares k sets of trainset and testset using cross-validation
    - Saves the datasets as npz file format

-----

## Experiment1 (Using original data)

### Source Codes

- `main_knn.py`
- `main_rf.py`
- `main_svm.py`
- `main_nn.py`
- `classification_algo.py`: Holds the classifications algorithms used in the experiment.

### Prerequisite

Since the codes are written in Python 3, if you run the code on the campus servers, turn Python 3 module on by running:

```
module load python/3.6.0
```

### Classifier 

I tried various sets of parameters for different classifiers. The result is saved here:

- knn.dat
- rf.dat
- svm.dat
- nn.dat

### Testing

- Parameter tuning for K nearest neighbor

    - number of neighbors: (3, 5, 7, ... , 19)
    - weights tuple: ("uniform", "distance")

```
python main_knn.py -i dataset.npz
```

- Parameter tuning for Random Forest

    - criterion tuple: ("gini", "entropy")
    - number of trees: 4, 8, 16, ... , 4096
    - minimum number of samples required to split an internal node: 2, 4, 8, ... , 32

```
python main_rf.py -i dataset.npz
```

- Parameter tuning for SVM

    - kernel: 'linear', 'poly', 'rbf', 'sigmoid'
    - penalty parameter C of the error term: 2^(-3), 2^(-1), ..., 2^(15)
    - gamma (Kernel coefficient for 'rbf', 'poly' and 'sigmoid'): 2^(-15), 2^(-13), ..., 2^(3)
    - degree of the polynomial kernel function: 1, 2, ... , 6

```
python main_svm.py -i dataset.npz -kernel linear
python main_svm.py -i dataset.npz -kernel poly
python main_svm.py -i dataset.npz -kernel rbf
python main_svm.py -i dataset.npz -kernel sigmoid
```

- Parameter tuning for Neural Network

    - hidden layers: 125 different hidden layers (2, 2, 2) to (32, 32, 32)125 different hidden layers (2, 2, 2) to (32, 32, 32)
    - activation function for the hidden layer: 
        - identity: f(x) = x
        - logistic: f(x) = 1 / (1 + exp(-x))
        - tanh: f(x) = tanh(x)
        - relu: f(x) = max(0, x)
    - solver for weight optimization:
        - 'lbfgs' is an optimizer in the family of quasi-Newton methods
        - 'sgd' ref    ers to stochastic gradient descent
        - 'adam' refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Ji    mmy
        - note: The default solver 'adam' works pretty well on relatively large datasets (with thousands of training samples or mo    re) in terms of both training time and validation score. For small datasets, however, 'lbfgs' can converge faster and perform b    etter. 
    - alpha(L2 penalty (regularization term) parameter): 2^(-14), 2^(-13), ... , 1

```
python main_nn.py -i dataset.npz -activation identity -solver lbfgs
python main_nn.py -i dataset.npz -activation identity -solver sgd
python main_nn.py -i dataset.npz -activation identity -solver adam
python main_nn.py -i dataset.npz -activation logistic -solver lbfgs
python main_nn.py -i dataset.npz -activation logistic -solver sgd
python main_nn.py -i dataset.npz -activation logistic -solver adam
python main_nn.py -i dataset.npz -activation tanh -solver lbfgs
python main_nn.py -i dataset.npz -activation tanh -solver sgd
python main_nn.py -i dataset.npz -activation tanh -solver adam
python main_nn.py -i dataset.npz -activation relu -solver lbfgs
python main_nn.py -i dataset.npz -activation relu -solver sgd
python main_nn.py -i dataset.npz -activation relu -solver adam
```

### Result



-----

## Experiment2
