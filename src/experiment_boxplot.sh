####################################################
# 

# Change the Python version
module load python/3.6.0

# Original data - Experiment with 4 different models
python src/main_knn3.py -i data/dataset.npz -n 7 -weights distance > result/boxplot_original.dat
python src/main_svm3.py -i data/dataset.npz -C 0.125 -kernel poly -gamma 0.03125 -degree 1 >> result/boxplot_original.dat
python src/main_rf3.py -i data/dataset.npz -criterion gini -n 1024 -minss 2 >> result/boxplot_original.dat
python src/main_nn3.py -i data/dataset.npz -hls 64 64 64 -activation tanh -solver lbfgs -alpha 0.5 >> result/boxplot_original.dat

# PCA dim reducted data - Experiment with 4 different models
python src/main_knn3.py -i data/dataset_pca100.npz -n 7 -weights distance > result/boxplot_pca.dat
python src/main_svm3.py -i data/dataset_pca50.npz -C 0.125 -kernel poly -gamma 0.03125 -degree 1 >> result/boxplot_pca.dat
python src/main_rf3.py -i data/dataset_pca50.npz -criterion gini -n 1024 -minss 2 >> result/boxplot_pca.dat
python src/main_nn3.py -i data/dataset_pca100.npz -hls 64 64 64 -activation tanh -solver lbfgs -alpha 0.5 >> result/boxplot_pca.dat
