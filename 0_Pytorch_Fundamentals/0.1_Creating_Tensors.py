import torch as t

# scalar (i hardly know 'er!) - 0 dimension tensor
scalar = t.tensor(5)
print(scalar, scalar.ndim, scalar.item()) # scalar, dimensions, scalar's item (item seems to only work for scalars)
# vector (the guy from despicable me) - 1 dim tensor
vector = t.tensor([5,6,7,8])
print(vector, vector.ndim, vector.shape) # 4 in shape
# matrix (that one long movie) - 2 dim tensor
matrix = t.tensor([[1,2,3],[4,5,6]])
print(matrix, matrix.ndim, matrix.shape) # 2x3 in shape
# tensor - can be any dimensions
tensor = t.tensor([[[10,11,12],[13,14,15]]]) # 1x2x3 in shape
print(tensor, tensor.ndim, tensor.shape)
# random tensor creation
rand_tensor = t.rand(size=(2,3))
print(rand_tensor)
# zeros and ones tensor creation
zeros_tensor = t.zeros(size=(1,2))
ones_tensor = t.ones(size=(1,2))
print(f"{zeros_tensor}\n{ones_tensor}")
# arrange - create a tensor same manner as range()
ranged_tensor = t.arange(0,10,2)
print(ranged_tensor)
# like - creates same shape
template = t.rand(size=(2,2))
copy = t.ones_like(template)
print(copy, copy.shape)
# Default datatype for tensors is float32
float_32_tensor = t.tensor([3.0, 6.0, 9.0],
                            dtype=t.float16, # datatype of elements - defaults to None, which is torch.float32
                            device=None, # what device tensor is stored - defaults to None
                            requires_grad=False) # if True, operations performed on the tensor are recorded 
print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)