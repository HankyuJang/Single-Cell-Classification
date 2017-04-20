module load python/3.6.0

python src/plot_best_four_boxplot.py -i result/boxplot_original.dat -k 5 -t Performance_comparison_original -o plots/boxplot_original.png
python src/plot_best_four_boxplot.py -i result/boxplot_pca.dat -k 5 -t Performance_comparison_PCA -o plots/boxplot_pca.png
