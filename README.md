# ukladybirds

The presented repository contains the main scripts of the work presented in the article entitled "".
It consists of 3 main folders:

## CNNs
The five scripts represents the five Convolutional Neural Network models presented in the article: ResNet50 with a 128 nodes dense layer, ResNet50 with a 256 nodes dense layer, ResNet152 with a 128 nodes dense layer, ResNet152 with a 256 nodes dense layer and ResNet50 with the classification head as presented in Terry et al., 2020.

## SDMs
The three scripts represents the three variations of the MaxEnt algorithm adopted in the article. "maxent_1000.R" with 1000 points of pseudoabsence, "maxent_400.R" with 400 points of pseudoabsence and "maxent_400_mr.R" with 400 points of pseudoabsence and multiple records per pixel.

## GAs
The scripts "ga_1_classic_tournament_paralell.py" and "ga_18_classic_tournament_paralell.py" are the genetic algorithms that optimize respectively 1 and 18 parameters. "predictions_for_ga.py" prepares the individual outputs for genetic algorithm use.
