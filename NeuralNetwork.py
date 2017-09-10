import numpy as np

class NeuralNetwork:
    def __init__(self, eons=10000, size=[3,1], seed=2017, verbose=False,
#                trainingInputs=np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]),
#                trainingOutput=np.array([[0, 1, 1, 1, 1, 0, 0]]).T):
                trainingInputs = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]),
                trainingOutput = np.array([[0,0,1,1]]).T):

        self.eons = eons
        self.size = size
        self.layers = len(size)
        self.trainingInput = trainingInputs
        self.trainingOutput = trainingOutput
        self.trainedWeights = None
        self.verbose = verbose
        np.random.seed(seed)

    def _sigmoid(self, x):
        # Converts weighted inputs to 0,1 scale
        return(1 / (1 + np.exp(-x)))

    def _sigDeriv(self, x):
        # Take gradient of sigmoid position
        # Extreme (near 0/1) changes are weighted less than mid (~0.5) changes
        return(x * (1 - x))

    def train(self):
        # Set initial weights as random
        weights = []
        for i in range(self.layers-1):
            weights.append(2 * np.random.random((self.size[i],self.size[i+1])) - 1)

        # Loop through network
        for _ in range(self.eons):
            # Take input and apply weights
            states = [self.trainingInput] # input, layerN, output

            # Calculate layer states
            for i in range(self.layers-1):
                states.append(self._sigmoid(np.dot(states[-1], weights[i])))

            # Calculate adjustments due to error, first is a little different
            errors = []
            weight_adjs = []
            errors.append(self.trainingOutput - states[-1])
            weight_adjs.append( (self.trainingOutput - states[-1]) * self._sigDeriv(states[-1]))
            if self.layers > 2:
                for i in range(1, self.layers-1):
                    errors.append(np.dot(weight_adjs[-1], weights[-i].T))
                    weight_adjs.append(errors[-1] * self._sigDeriv(states[i]))

            #print('Weights:', weights) if self.verbose else None
            #print('States:', states) if self.verbose else None

            errors      = errors[::-1]
            weight_adjs = weight_adjs[::-1]

            #print('Errors:', errors) if self.verbose else None
            #print('Weights Adj:', weight_adjs) if self.verbose else None

            # Apply weight adjustments
            for i in range(self.layers-1):
                weights[i] += np.dot(states[i].T, weight_adjs[i])
            #print('Adjusted Weights:', weights) if self.verbose else None
        self.trainedWeights = weights
        return(weights)

    def test(self, testIn):
        assert self.trainedWeights != None, 'Error: Model must be trained first.'
        testOut = testIn
        for weights in self.trainedWeights:
            testOut = self._sigmoid(np.dot(testOut, weights))
        return(float(testOut))
