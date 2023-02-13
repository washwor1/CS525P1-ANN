import numpy as np
import sys
"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):
        self.func = activation
        self.value = input_num
        self.learning_rate = lr
        
        if weights:
            self.weights = weights
            
        else:
            self.weights = np.random.random()    
        
    #This method returns the activation of the net
    def activate(self, net):
        if self.func == 1:
            return 1 / (1 + np.exp(-net))
        
        elif self.func == 0:
            return net
        
        else:
            raise ValueError('Invalid activation function')      
        
    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        self.inputs = input
        self.net = np.dot(input, self.weights)
        self.output = self.activate(self.net)
        return self.output

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        if self.func == 1:
            return self.output * (1 - self.output)
        
        elif self.func == 0:
            return 1
        
        else:
            raise ValueError('Invalid activation function')

          
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        self.delta = wtimesdelta * self.activationderivative()
        self.partial_derivatives = np.dot(np.transpose(self.inputs), self.delta)
        return np.dot(self.delta, self.weights)


         
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        self.weights -= self.learning_rate * self.partial_derivatives


        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self, numOfNeurons, activation, input_num, lr, weights=None):
        self.neurons = []
        self.numOfNeurons = numOfNeurons
        self.input_num = input_num
        
        for i in range(numOfNeurons):
            if weights:
                self.neurons.append(Neuron(activation, input_num, lr, weights[i]))
            else:
                self.neurons.append(Neuron(activation, input_num, lr))
        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        outputs = []
        
        for neuron in self.neurons:
            output = neuron.calculate(input)
            outputs.append(output)
            
        return np.array(outputs)

        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        partial_derivatives = []
        
        for neuron in self.neurons:
            partial_derivative = neuron.calcpartialderivative(wtimesdelta)
            partial_derivatives.append(partial_derivative)
        
        sum_of_w_delta = np.sum(partial_derivatives, axis=0)
        
        for neuron in self.neurons:
            neuron.updateweights()
            
        return sum_of_w_delta

           
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None):
        self.numOfLayers = numOfLayers
        self.numOfNeurons = numOfNeurons
        self.inputSize = inputSize
        self.activation = activation
        self.loss = loss
        self.lr = lr
        self.layers = []
        
        for i in range(numOfLayers):
            if i == 0:
                input_num = self.inputSize
            else:
                input_num = self.numOfNeurons[i-1]
                
            if weights is not None:
                layer_weights = weights[i]
            else:
                layer_weights = None
                
            layer = FullyConnected(self.numOfNeurons[i], self.activation[i], input_num, self.lr, layer_weights)
            self.layers.append(layer)

    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self, input):
        current_input = input
        for layer in self.layers:
            current_input = layer.calculate(current_input)
        return current_input

        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self, yp, y, loss_function=0):
        if loss_function == 0:
            loss = np.sum((yp - y)**2) / 2
        elif loss_function == 1:
            loss = -np.mean(y * np.log(yp) + (1 - y) * np.log(1 - yp))
        else:
            raise ValueError("Invalid loss function.")
        return loss

    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self, yp, y):
        if self.loss == 0:
            return 2 * (yp - y)
        elif self.loss == 1:
            return yp - y
        else:
            raise ValueError("Invalid loss function")

    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self, x, y):
        # forward pass to calculate the predicted output
        yp = self.calculate(x)
        
        # calculate the loss
        loss = self.calculateloss(yp, y)
        print(loss)
        
        # calculate the derivative of the loss
        delta = self.lossderiv(yp, y)
        
        # initialize wtimesdelta with delta
        wtimesdelta = delta
        
        # iterate over all layers in reverse order
        for i in range(self.numOfLayers-1, -1, -1):
            # call calcwdeltas on the current layer with the wtimesdelta from the previous layer
            wtimesdelta = self.layers[i].calcwdeltas(wtimesdelta)


if __name__=="__main__":
    if (len(sys.argv)<2):
        print('a good place to test different parts of your code')
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1])
        np.array([0.01,0.99])
        
