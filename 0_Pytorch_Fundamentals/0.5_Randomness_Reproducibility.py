import torch as t

# random tensors
rand_A = t.rand(3,3)
rand_B = t.rand(3,3)
print(rand_A==rand_B) # since theyre random, unlikely to be = anywhere
# seed settings (like minecraft! Wow!)
t.manual_seed(seed=42) # sets seed for next rand
rand_C = t.rand(3,3)
t.manual_seed(seed=42)
rand_D = t.rand(3,3)
print(rand_C==rand_D) # now theyre the same!