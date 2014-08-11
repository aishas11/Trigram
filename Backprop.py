import theano
import theano.tensor as T
import numpy as np 
import Layer 

class Backprop(object): 
	''' 
	Main Backprop class: computes the outputs of a sequence of Layers, the associated crossEntropy error, 
	and the resulting gradients for weight/bias updates
	- names : these are the names of the layers, important for computing layer_ouputs individually
		most likely: ["layer_h1", "layer_h2", "layer_out"]
	- W_inits : list of np.ndarray, len=N
    	Values to initialize the weight matrix in each layer to.
        The layer sizes will be inferred from the shape of each matrix in W_init
    - b_inits : list of np.ndarray, len=N
        Values to initialize the bias vector in each layer to
    - activations : list of theano.tensor.elemwise.Elemwise, len=N
        Activation function for layer output for each layer
    '''

	def __init__(self, names, W_inits, b_inits, activations): 
		self.layers = []
		for name, W, b, activation in zip(names, W_inits, b_inits, activations): 
			self.layers.append(Layer.Layer(name, W, b, activation))
		self.params = []
		for layer in self.layers: # self.params = [W_inp_hid1, W_hid1_hid2, b_hid2, W_hid2_out, b_out]
			self.params += layer.params

	# uses the Layer.layer_output() function to recursively compute layer outputs and final network output 
	def network_output(self, in_batch): 
		layer_activation_outputs = []
		x = in_batch
		for layer in self.layers: 
			x = layer.layer_output(x)
			layer_activation_outputs.append(x)
		return layer_activation_outputs

	# calculate cross entropy error based on current network state 
	def calculate_CE_error(self, in_batch, out_batch_flat): 
		output_layer_state = self.network_output(in_batch)[-1]
		tiny = np.exp(-30)
		return -T.sum(out_batch_flat * T.log(output_layer_state + tiny)) / T.cast(in_batch.shape[0], theano.config.floatX)

	# calculate error gradients in order to update new weight/bias values 
	def calculate_gradients(self, in_batch, identity_matrix, out_batch_flat): 
		layer_outputs = self.network_output(in_batch)
		error_deriv = layer_outputs[-1] - out_batch_flat
		w_h2_out_gradient = T.dot(layer_outputs[1].T, error_deriv)
		b_out_gradient = T.sum(error_deriv, axis={0}, keepdims=True)
		bprop_deriv_1 = T.dot(error_deriv, self.layers[-1].getWeights().T) * layer_outputs[1] * (1 - layer_outputs[1])
		w_h1_h2_gradient = T.dot(layer_outputs[0].T, bprop_deriv_1)
		b_h2_gradient = T.sum(bprop_deriv_1, axis={0}, keepdims=True)
		bprop_deriv_2 = T.dot(bprop_deriv_1, self.layers[1].getWeights().T)
		num_vocab_words = self.layers[0].getWeights().shape[0]
		h1_numUnits = self.layers[0].getWeights().shape[1]
		w_inp_h1_gradient = T.tensordot(
			identity_matrix[in_batch.flatten()-1].reshape((in_batch.shape[0], in_batch.shape[1], num_vocab_words)),
			bprop_deriv_2.reshape((in_batch.shape[0], in_batch.shape[1], h1_numUnits)), 
			axes=([0,1],[0,1]))
		return [w_inp_h1_gradient, w_h1_h2_gradient, b_h2_gradient, w_h2_out_gradient, b_out_gradient]

