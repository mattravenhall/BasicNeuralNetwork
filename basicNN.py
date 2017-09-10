import numpy as np

def sigmoid(x):
    # Converts weighted inputs to 0,1 scale
    return(1 / (1 + np.exp(-x)))

def sigDeriv(x):
    # Take gradient of sigmoid position
    # Extreme (near 0/1) changes are weighted less than mid (~0.5) changes
    return(x * (1 - x))

class TwoLayerNN:
    trainingInputs = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    trainingOutput = np.array([[0,0,1,1]]).T

    def train(self, trainIn=trainingInputs, trainOut=trainingOutput, seed=2017, eons=10000):
        # Set seed for replication
        np.random.seed(seed)

        # Set initial weights as random
        weights_A = 2 * np.random.random((3,1)) - 1 # 3 input nodes to 1 output node

        # Loop through network
        for _ in range(eons):
            # Take input and apply weights
            output = sigmoid(np.dot(trainIn, weights_A))

            # Update weights according to error, also null weights on 0 inputs
            output_error = trainOut - output
            weights_A += np.dot(trainIn.T, (output_error) * sigDeriv(output))
        return(weights_A)

    def test(self, testIn, trainedWeights):
        testIn = np.array(testIn)
        return(float(sigmoid(np.dot(testIn, trainedWeights))))

class ThreeLayerNN:
    trainingInputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    trainingOutput = np.array([[0, 1, 1, 1, 1, 0, 0]]).T
    #trainingInputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    #trainingOutput = np.array([[0,0,1,1]]).T

    def train(self, trainIn=trainingInputs, trainOut=trainingOutput, seed=2017,
              eons=10000, verbose=False):
        # Set seed for replication
        np.random.seed(seed)

        # Set initial weights as random
        weights_A = 2 * np.random.random((3,4)) - 1 # 3 input nodes to 4 hidden nodes
        weights_B = 2 * np.random.random((4,1)) - 1 # 4 hidden nodes to 1 output node
        #print('Weights:', [weights_A, weights_B]) if verbose else None

        # Loop through network
        for _ in range(eons):
            # Take input and apply weights from each layer
            output_A = sigmoid(np.dot(trainIn, weights_A))
            output_B = sigmoid(np.dot(output_A, weights_B))

            # Determine error & adjustments needed for weights
            output_B_error = trainOut - output_B
            weights_B_adj = (output_B_error) * sigDeriv(output_B)
            output_A_error = np.dot(weights_B_adj, weights_B.T)
            weights_A_adj = (output_A_error) * sigDeriv(output_A)

            # Update weights accordingly
            weights_A += np.dot(trainIn.T, weights_A_adj)
            weights_B += np.dot(output_A.T, weights_B_adj)

        #print('States:', [output_A, output_B]) if verbose else None
        #print('Errors:', [output_A_error, output_B_error]) if verbose else None
        print('Weights Adj:', [weights_A_adj, weights_B_adj]) if verbose else None

        return([weights_A, weights_B])

    def test(self, testIn, trainedWeights):
        testIn = np.array(testIn)
        layerOneOut = sigmoid(np.dot(testIn, trainedWeights[0]))
        layerTwoOut = sigmoid(np.dot(layerOneOut, trainedWeights[1]))
        return(float(layerTwoOut))
