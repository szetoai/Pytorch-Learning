import torch as t

t.set_default_device("cuda")
# 2. Create a random tensor with shape (7, 7).
rand_tens = t.rand(size=(7,7))
# 3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) (hint: you may have to transpose the second tensor).
mult_tens = t.rand(size=(1,7))
mult_result = rand_tens.mm(mult_tens.transpose(0,1))
# 4. Set the random seed to 0 and do 2 & 3 over again.
t.cuda.manual_seed(0) # gpu equivalent of t.manual_seed
rand_tens = t.rand(size=(7,7))
t.manual_seed(0)
mult_tens = t.rand(size=(1,7))
mult_result = rand_tens.mm(mult_tens.transpose(0,1))
# 5. Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? If there is, set the GPU random seed to 1234.
# 6. Create two random tensors of shape (2, 3) and send them both to the GPU. Set torch.manual_seed(1234) when creating the tensors.
t.cuda.manual_seed(1234)
p6_tens1 = t.rand(size=(2,3))
t.cuda.manual_seed(1234)
p6_tens2 = t.rand(size=(2,3))
# 7. Perform a matrix multiplication on the tensors you created in 6.
p7 = p6_tens1.mm(p6_tens2.transpose(0,1))
# 8. Find the maximum and minimum values of the output of 7.
p8 = (t.min(p7), t.max(p7))
# 9. Find the maximum and minimum index values of the output of 7.
p9 = (t.argmin(p7), t.argmax(p7))
'''
10. Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10). 
Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.
'''
t.cuda.manual_seed(7)
p10_rand = t.rand(size=(1,1,1,10))
print(p10_rand, p10_rand.shape)
p10_squeeze = t.squeeze(p10_rand)
print(p10_squeeze, p10_squeeze.shape)