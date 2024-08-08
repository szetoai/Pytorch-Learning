import torch as t

print(t.cuda.is_available(), t.cuda.device_count()) # true - gpu can be used
# set device to whichever is usable
device_choice = "cuda" if t.cuda.is_available() else "cpu"
# moving tensors to GPU
cpu_tensor = t.tensor([1,2,3])
gpu_tensor = cpu_tensor.to(device_choice)
gpu_to_cpu = gpu_tensor.cpu()
print(cpu_tensor) # cpu
print(gpu_tensor) # gpu
print(gpu_to_cpu) # cpu