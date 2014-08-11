import os, csv, sys
import numpy as np 
import pylab as Plot

import theano
import theano.tensor as T

import tsne
import Backprop
import FileReader

# use tsne - t-Distributed Stochastic Neighbor Embedding - 
# to illustrate learned distributed representations of words in given dictionary
# (see: http://homepage.tudelft.nl/19j49/t-SNE.html)
show_word_neighbors = True

# training arguments
numIterations = 100 # how many times to go through the training set 
input_batch_size = 1000 # how many trigrams to present at a time

# hyperparameters
LR = 0.1 
momentum = 0.9 

# initialize layer sizes 
num_distributed_hid1 = 50  # look-up table/distributed representation mapping
num_hid2 = 200  
num_vocab_words = 250 # output layer size 

# training, validation and test data
filenames = ['Train_Sequences.csv', 'Validation_Sequences.csv', 'Test_Sequences.csv']
dataFiles_Path = os.path.join(os.getcwd(), 'data_files/')
inputDataArrays_List = []
targetDataArrays_List = []
for fn in filenames: 
	fr = FileReader.FileReader(os.path.join(dataFiles_Path,fn))
	npdata = fr.csv_to_npArray_ints()
	inputDataArrays_List.append(npdata[:,:-1])
	targetDataArrays_List.append(npdata[:,-1:])

# calculate sizes of weights and biases
batch_Inps = T.imatrix('batch_Inps')
batch_Outs_Desired = T.imatrix('batch_Outs_Desired')
totalBatches = inputDataArrays_List[0].shape[0] / input_batch_size + 1
inp_seqLength = inputDataArrays_List[0][0].shape[0]
flatMat_eye = theano.shared(np.eye(num_vocab_words, dtype=theano.config.floatX))
batch_Outs_Desired_Flat = flatMat_eye[batch_Outs_Desired.flatten()-1]

# calculate sizes of weights and biases
weights_sizes = [(num_vocab_words,num_distributed_hid1), (num_distributed_hid1*inp_seqLength,num_hid2), (num_hid2,num_vocab_words)] # [(250,50),  (150,200),  (200,250)]
bias_sizes = [(1,num_hid2),(1,num_vocab_words)]

# initialize inputs and instantiate backprop class 
np.random.seed(234) 
layer_names = ["layer_h1", "layer_h2", "layer_out"]
weights_stdev = 0.01 
W_initial = [0.01 * np.random.randn(ws[0], ws[1]) for ws in weights_sizes]
b_initial = [None] + [np.zeros((bs[0], bs[1])) for bs in bias_sizes]
act_initial = [None, T.nnet.sigmoid, T.nnet.softmax]
bp = Backprop.Backprop(layer_names, W_initial, b_initial, act_initial)
gradients_List = bp.calculate_gradients(batch_Inps, flatMat_eye, batch_Outs_Desired_Flat)
weights_biases_List = bp.params 

# declare function that calculates new delta weight/bias updates 
update_deltas = []
deltas = []
for gradient, param in zip(gradients_List, weights_biases_List): 
	delta = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
	update_deltas.append(( delta, (momentum * delta + gradient) / T.cast(batch_Outs_Desired.shape[0], theano.config.floatX) ))
	deltas.append(delta)
train = theano.function(inputs=[batch_Inps, batch_Outs_Desired], 
	outputs = bp.calculate_CE_error(batch_Inps, batch_Outs_Desired_Flat), 
	updates = update_deltas ) 

# declare function that calculates new weight/bias updates 
update_weights_biases = []
for parameter, delta_update in zip(weights_biases_List, deltas): 
	update_weights_biases.append(( parameter, parameter - LR * delta_update ))
update_W_B = theano.function(inputs=[], outputs=[], updates = update_weights_biases)

# loop through all the batches for numIterations, and run backpropagation model on these inputs 
for iteration in range(numIterations):
	print "Iteration " + str(iteration)
	for i in range(totalBatches):
		CE = float(train(inputDataArrays_List[0][i * input_batch_size : (i+1) * input_batch_size,:],
			targetDataArrays_List[0][i * input_batch_size : (i+1) * input_batch_size,:]))
		update_W_B()
		print "Batch " + str(i) + " CE = " + str(CE)

# Validation and test sets 
Validation_CE = float(train(inputDataArrays_List[1], targetDataArrays_List[1]))
Test_CE = float(train(inputDataArrays_List[2], targetDataArrays_List[2]))
print "Final Validation CE = " + str(Validation_CE)
print "Final Test CE = " + str(Test_CE)

# if you want to show nearest words 
if show_word_neighbors: 
	w_inp_hid1_final = weights_biases_List[0].get_value()
	vocab_words = FileReader.FileReader(os.path.join(dataFiles_Path,'vocab.csv')).csv_1RowStr_to_list()
	mappedX = tsne.tsne(w_inp_hid1_final)
	fig = Plot.figure(figsize=(15,8),dpi=100)
	for a, b, label in zip(mappedX[:,0],mappedX[:,1],vocab_words): 
		Plot.text(a,b,label)
	Plot.title('Distributed Word Representation Learned by Network - words spatially close also have similar distributed representations in the model')
	Plot.axis([-25, 25, -25, 25])
	Plot.subplots_adjust(left=.05, bottom=.05, right=.95, top=.95, wspace=.2, hspace=.2)
	Plot.savefig('fig1.jpg', dpi=100)
	Plot.show()


