import theano
import theano.tensor as T
import numpy as np 

class Layer(object): 

	def __init__(self, name, W_init, b_init, activation):
		'''
        A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector
        	- name : string - either "layer_h1", "layer_h2", "layer_out" 
        		This is important for calculating the layer_output, (see layer_output function)
            - W_init : np.ndarray, shape=(n_output, n_input)
                Values to initialize the weight matrix to.
            - b_init : np.ndarray, shape=(n_output,)
                Values to initialize the bias vector
            - activation : theano.tensor.elemwise.Elemwise
                Activation function for layer output
        '''
		self.name = name
		self.W = theano.shared(value=W_init.astype(theano.config.floatX))
		self.b = theano.shared(value=b_init.astype(theano.config.floatX), broadcastable=(True,False)) if b_init!=None else None
		self.activation = activation if activation else None
		self.params = [self.W, self.b] if b_init!=None else [self.W]

	# function for calculating the output state of a network layer 
	def layer_output(self, layerInp, hid1_numUnits = 50): 
		if self.name == "layer_h1": 
			returnVal = T.reshape(self.W[layerInp.flatten()-1], (layerInp.shape[0], layerInp.shape[1] * hid1_numUnits))
		elif self.name == "layer_h2": 
			linOut = T.dot(layerInp, self.W) + self.b
			returnVal = self.activation(linOut)
		elif self.name == "layer_out": 
			linOut = T.dot(layerInp, self.W) + self.b
			returnVal = self.activation(linOut - T.max(linOut))
		return returnVal

	def getWeights(self): 
		return self.W