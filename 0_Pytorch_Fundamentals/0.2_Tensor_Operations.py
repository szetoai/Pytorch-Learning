import torch as t

# element-wise operations - applies operations to all elements
math_tensor = t.ones(size=(3,3))
print(f"{math_tensor * 10}\n{ math_tensor + 5}\n{math_tensor - 3}\n{math_tensor/16}")
# matrix multiplication/dot product - inner dimensions MUST match (i.e 3x3 * 3x2 = 3x2 (outer dimensions))
matrix = t.tensor([[1,2,3],[4,5,6],[7,8,9]])
matrix1 = t.tensor([[10,9],[8,7],[6,5]])
print(t.matmul(matrix,matrix1)) # matmul = mm for short (for mm, its self.mm(other matrix))
# transposing - swapping dimensions of tensor to make mm work
bad_matrix = t.tensor([[1,2,3],[4,5,6]])
bad_matrix1 = t.tensor([[7,8,9],[10,11,12]]) 
# 2x3 * 2x3 doesnt work! so...
bad_matrix1 = t.transpose(bad_matrix1, 0, 1) # transpose(input, dim1, dim2) swaps dim1/dim2
# now, bad_matrix1 is 3x2, to mm works!
bad_matrix = bad_matrix.mm(bad_matrix1)
print(bad_matrix)

# aggregation - going from more -> less elements
combine = t.arange(0,10,1)
print(t.min(combine),t.max(combine),t.mean(combine,dtype=float),t.sum(combine),combine.argmin(),combine.argmax()) # mean must be in float, argmin/max returns index of min/max
# changing data types
combine16 = combine.type(t.float16)
print(combine16.dtype)
# reshaping - t.reshape(input, shape) - use view(shape) to just see it
combine, combine16 = t.reshape(combine, (2,5)), t.reshape(combine16, (2,5))
# permuting - similar to reshape, but changes order of elements
print(t.permute(combine, (1,0))) # 2nd dim size is now 1st, 1st dim size is now in 2nd place
# stacking - concatenates tensors along a dimension (must be same size)
stack = t.stack((combine, combine16), dim=1) # you can stack with yourself!
print(stack)
# squeezing - removes all dimensions of size one 
orange = t.ones(size=(3,1,4,1,5))
juice = t.squeeze(orange) # 3x1x4x1x5 -> 3x4x5
print(orange, orange.shape, juice, juice.shape)
# unsqueezing - adds dimension at an index
juice = t.unsqueeze(juice, dim=1) # 3x4x5 -> 3x1x4x5
print(juice.shape)

# indexing - pretty similar to lists
tens_index = t.arange(0,30,1).reshape(2,5,3)
print(tens_index[1,:,2]) # : for all