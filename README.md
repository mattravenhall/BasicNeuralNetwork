# BasicNeuralNetwork
Just messing around with neural network fundamentals

## Running basicNN
- Basic neural networks of specific sizes (two or three layer), mainly learn how to build one.

```python
>>> import basicNN as bNN
>>> n = bNN.TwoLayerNN()
>>> w = n.train()
>>> n.test([1,0,0],w)
0.9999370428352157
```

## Running NeuralNetwork
- Combined and extended version of basicNN, allowing for creating neural networks of any specific size.

```python
>>> import NeuralNetwork as NN
>>> n = NN.NeuralNetwork(size=[3,1], eons=10000)
>>> n.train(trainingInputs=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]), trainingOutput=np.array([[0,0,1,1]]).T))
>>> n.test([1,0,0])
0.9999370428352157
```
