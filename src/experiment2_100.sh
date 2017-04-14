####################################################
# PCA using 100 basis vectors

# Change the Python version
module load python/3.6.0

# Create datasets
python src/data_cleaning_pca.py -i data/expressionmRNAAnnotations.txt -o data/dataset_pca100 -kfold 5 -n 100

# Experiment with 8 different models
python src/main_svm2.py -i data/dataset_pca100.npz -C 0.125 -kernel linear > result/pca100.dat
python src/main_svm2.py -i data/dataset_pca100.npz -C 0.125 -kernel poly -gamma 0.03125 -degree 1 >> result/pca100.dat
python src/main_knn2.py -i data/dataset_pca100.npz -n 7 -weights uniform >> result/pca100.dat
python src/main_knn2.py -i data/dataset_pca100.npz -n 7 -weights distance >> result/pca100.dat
python src/main_nn2.py -i data/dataset_pca100.npz -hls 64 64 64 -activation tanh -solver lbfgs -alpha 0.5 >> result/pca100.dat
python src/main_nn2.py -i data/dataset_pca100.npz -hls 32 64 64 -activation relu -solver lbfgs -alpha 0.00048828125 >> result/pca100.dat
python src/main_rf2.py -i data/dataset_pca100.npz -criterion gini -n 1024 -minss 2 >> result/pca100.dat
python src/main_rf2.py -i data/dataset_pca100.npz -criterion entropy -n 1024 -minss 4 >> result/pca100.dat


