####################################################
# PCA using 7 basis vectors

# Change the Python version
module load python/3.6.0

# Create datasets
python src/data_cleaning_pca.py -i data/expressionmRNAAnnotations.txt -o data/dataset_pca7 -kfold 5 -n 7

# Experiment with 4 different models
python src/main_svm2.py -i data/dataset_pca7.npz -C 0.125 -kernel poly -gamma 0.001953125 -degree 1 > result/pca7.dat
python src/main_knn2.py -i data/dataset_pca7.npz -n 3 -weights uniform >> result/pca7.dat
python src/main_nn2.py -i data/dataset_pca7.npz -hls 32 32 64 -activation identity -solver adam -alpha 0.00390625 >> result/pca7.dat
python src/main_rf2.py -i data/dataset_pca7.npz -criterion gini -n 128 -minss 4 >> result/pca7.dat
