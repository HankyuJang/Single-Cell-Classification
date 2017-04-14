########################################################################
# Experiment 1
python src/plot_best_four.py -i result/best_results.dat

########################################################################
# Experiment 2
python src/plot_best_four2.py -i result/pca7.dat -o plots/pca7.png -n 7
python src/plot_best_four2.py -i result/pca13.dat -o plots/pca13.png -n 13
python src/plot_best_four2.py -i result/pca25.dat -o plots/pca25.png -n 25
python src/plot_best_four2.py -i result/pca50.dat -o plots/pca50.png -n 50
python src/plot_best_four2.py -i result/pca100.dat -o plots/pca100.png -n 100
python src/plot_best_four2.py -i result/pca200.dat -o plots/pca200.png -n 200
python src/plot_best_four2.py -i result/pca400.dat -o plots/pca400.png -n 400
python src/plot_best_four2.py -i result/pca500.dat -o plots/pca500.png -n 500
python src/plot_best_four2.py -i result/pca800.dat -o plots/pca800.png -n 800
python src/plot_best_four2.py -i result/pca1600.dat -o plots/pca1600.png -n 1600
