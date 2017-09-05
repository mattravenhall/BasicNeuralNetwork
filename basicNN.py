import numpy as np

class SimpleNN:
    trainingInputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    trainingOutput = np.array([[0,1,1,0]]).T

    def sigmoid(self, x):
        # Converts weighted inputs to 0,1 scale
        return(1 / (1 + np.exp(-x)))

    def sigDeriv(self, x):
        # Take gradient of sigmoid position
        # Extreme (near 0/1) changes are weighted less than mid (~0.5) changes
        return(x * (1 - x))

    def trainModel(self, trainIn=trainingInputs, trainOut=trainingOutput, seed=2017, eons=10000):
        # Set seed for replication
        np.random.seed(seed)

        # Set initial weights as random
        weights = 2 * np.random.random((3,1)) - 1 # 3 input nodes to 1 output node

        # Loop through network
        for _ in range(eons):
            # Take input and apply weights
            output = self.sigmoid(np.dot(trainIn, weights))

            # Update weights according to error, also null weights on 0 inputs
            weights += np.dot(trainIn.T, (trainOut - output)*self.sigDeriv(output))

        return(weights)

    def testModel(self, testIn, trainedWeights):
        testIn = np.array(testIn)
        return(float(self.sigmoid(np.dot(testIn, trainedWeights))))
