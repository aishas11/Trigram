This repository is a theano implementation of the following Trigram Backprop Assignment: http://www.cs.toronto.edu/~nitish/csc321/assignment1.html
It trains a network that takes a sequence of words as input and learns to predict the word that comes next. 

Python Library Requirements: theano, numpy, pylab, csv
(should be able to do a pip install for all) 

How to run: 
python Trigram_Backprop_Main.py 


Trigram_Backprop_Main.py calls the Backprop class to calculate network output, CE_error, and gradients. Backprop class calls the Layer class to compute activation patterns for 3 layers (h1, h2, output) 

Classes:
Layer.py - computes layer outputs, s(Wx + b) where s is activation function and x is the input vector
Backprop.py - computes network output, cross entropy error, and parameter gradients 
FileReader.py - functions to parse csv files into various types of lists/arrays


tsne.py (not a class) - contains functions useful for illustrating learned distributed representations of words in this data set (see sample output/fig1.jpg for example)
more info on tsne here <http://homepage.tudelft.nl/19j49/t-SNE.html>