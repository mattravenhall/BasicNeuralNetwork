import NeuralNetwork as NN
import basicNN as bNN

seed = 1
verbose = False
eons = 10000
size = [3,1]
test = [1,0,0]

n = NN.NeuralNetwork(size=size, eons=eons, seed=seed, verbose=verbose)
n.train()
print('NN', n.test(test))

n2 = bNN.TwoLayerNN()
w = n2.train(eons=eons, seed=seed) #, verbose=verbose)
print('2L', n2.test(test, w))

#n3 = bNN.ThreeLayerNN()
#w = n3.train(eons=eons, seed=seed, verbose=verbose)
#print('3L', n3.test(test, w))
