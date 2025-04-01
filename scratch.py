import numpy as np
import logging

logger = logging.getLogger(__name__)

class Neuron:
    def __init__(self, inputs, weights, bias):
        self.weights = weights
        self.bias = bias
        self.inputs = inputs

    def sigmoid(self, x):
        # activation function: f(x) = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))

    def dot_product(self):

        output = sum([factor * weight for factor, weight in zip(self.inputs, self.weights)])
        return output

    def feedfoward(self):
        return self.sigmoid(self.dot_product() + self.bias)

class NeuralNetwork(Neuron):
    def __init__(self, num_layers, nodes_per_layer):

        super().__init__(inputs, weights, bias)
        self.weights = weights
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.bias = bias
        self.nodes = {}
        self.generate_nodes()


    def generate_nodes(self):
        for x, layer in enumerate(range(self.num_layers)):
            for i, node in enumerate(range(self.nodes_per_layer)):
                self.nodes[f'{x}_{i+1}'] = Neuron(np.array(np.random.rand(2, 1)), weights, bias)

    def train_network(self):

        """Recursive algorithm to feedforward through each layer
            until hitting the output """

        output = self.feedfoward()
        # base case
        if output == self.inputs * 0.9:
            return output
        # recursively travel through layers
        else:
            self.train_network()


inputs = np.array(np.random.rand(2, 1))
weights = np.random.rand(2, 1)
bias = 5

Network = NeuralNetwork(num_layers=5, nodes_per_layer=3)
Network.train_network()


# node1 = Neuron(inputs, weights, bias)
# node2 = Neuron(inputs, weights, bias)
first_node_output = node1.feedfoward()
second_node_output = node2.feedfoward()
output_node = Neuron(np.array([first_node_output, second_node_output]), weights, bias)
output_node_output = output_node.feedfoward()
print(output_node_output)
