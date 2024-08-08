import torch as t

'''
y = x * A^T + b
y - output
x - input
A - weights matrix that gets adjusted to represent data
T - weight matrix gets transposed
b - bias term
'''

rand = t.rand(size=(4,3))
linear = t.nn.Linear(in_features=3,out_features=7) # inner dim, outer dim of output
output = linear(rand)
print(f"{rand, rand.shape}\n{output, output.shape}")