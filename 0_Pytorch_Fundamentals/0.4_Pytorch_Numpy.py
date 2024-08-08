import torch as t
import numpy as np

# convert from array to tensor and back
array = np.array([1,2,3])
tensor = t.from_numpy(array) + 1
back = t.Tensor.numpy(tensor) + 1
print(array, tensor, back)