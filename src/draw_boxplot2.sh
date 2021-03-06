module load python/3.6.0

python src/main_knn3.py -i data/dataset_pca50.npz -n 3 -weights uniform > result/boxplot_final.dat
python src/main_svm3.py -i data/dataset.npz -C 0.125 -kernel poly -gamma 0.001953125 -degree 1 >> result/boxplot_final.dat
python src/main_rf3.py -i data/dataset.npz -criterion gini -n 128 -minss 4 >> result/boxplot_final.dat
python src/main_nn3.py -i data/dataset.npz -hls 32 32 64 -activation identity -solver adam -alpha 0.00390625 >> result/boxplot_final.dat

python src/plot_best_four_boxplot.py -i result/boxplot_final.dat -k 5 -t Performance_comparison_final -o plots/boxplot_final.png
