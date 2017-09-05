# BasicNeuralNetwork
Just messing around with neural network fundamentals

## Running basicNN

1. Import module
2. Create an instance of the SimpleNN class
3. Train model on included training dataset, assign weights to w
4. Pass these weights and test data to testModel

```python
>>> import basicNN as bNN
>>> n = bNN.SimpleNN()
>>> w = n.trainModel()
>>> n.testModel([1,0,0],w)
0.9999370101226006
```
