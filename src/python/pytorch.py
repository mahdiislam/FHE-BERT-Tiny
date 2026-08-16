import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchviz import make_dot
from sklearn.linear_model import LinearRegression

# Data Generation
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)

# Shuffles the indices
idx = np.arange(100)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:80]
# Uses the remaining indices for validation
val_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
# and then we send them to the chosen device
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

# Here we can see the difference - notice that .type() is more useful
# since it also tells us WHERE the tensor is (device)
# print(len(x_train), type(x_train), type(x_train_tensor), x_train_tensor.type(), len(x_train_tensor.numpy()))

#for i in range(len(x_train_tensor)):
#    print(x_train[i], x_train_tensor[i].numpy(), x_train[i] - x_train_tensor[i].numpy())

lr = 1e-1
n_epochs = 1000

torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

for epoch in range(n_epochs):
    # Step 1
    yhat = a + b * x_train_tensor
    # Step 2
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()     
    # Step 3
    # We just tell PyTorch to work its way BACKWARDS from the specified loss!
    loss.backward()
    # We need to use NO_GRAD to keep the update out of the gradient computation
    # Why is that? It boils down to the DYNAMIC GRAPH that PyTorch uses...
    # Step 4
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
    
    # PyTorch is "clingy" to its computed gradients, we need to tell it to let it go...
    a.grad.zero_()
    b.grad.zero_()
    
print(a, b)
