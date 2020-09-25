from __future__ import print_function
import torch
import numpy as np

y = torch.empty(5,3)
x = torch.rand(5,3) 
d = torch.zeros(5,3, dtype=torch.long)
t = torch.tensor([5.5, 3])
u = torch.randn_like(x, dtype=torch.float)

print(y)
print(x)
print(d)
print(t)
print(u)
print(u.size())
print(x + d)
print(torch.add(x,d))
print(x[:,1])

print('Is CUDA avaiable?', torch.cuda.is_available())

a = torch.ones(5)
print('tensor version', a)

b = a.numpy()
print('numpy version ', b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# use tensors to move in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda") # get object
    y = torch.ones_like(x, device=device) # create tensor on GPU
    x = x.to(device) # or just use strings .to cuda
    z = x + y
    print(z)
    print(z.to("cpu", torch.double)) # .to can change dtype

# autograd tracks all operations on tensors if set to True

x = torch.ones(2,2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

out.backward()
print(x.grad)

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)


print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())