import numpy as np
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
x_val, y_val = x[val_idx], y[val_idx]

# Initializes parameters "a" and "b" randomly
np.random.seed(42)
a = np.random.randn(1)  # Line 3
b = np.random.randn(1)  # Line 4

# print(a, b)

# Sets learning rate
lr = 1e-1               # Line 9
# Defines number of epochs
n_epochs = 1000         # Line 11

for epoch in range(n_epochs):
    # Step 1: Computes our model's predicted output
    yhat = a + b * x_train      # Line 15
    # Step 2
    # How wrong is our model? That's the error! 
    error = (y_train - yhat)    # Line 18
    # It is a regression, so it computes mean squared error (MSE)
    loss = (error ** 2).mean()  # Line 20
    # Step 3
    # Computes gradients for both "a" and "b" parameters
    a_grad = -2 * error.mean()             # Line 23
    b_grad = -2 * (x_train * error).mean() # Line 24
    # Step 4
    # Updates parameters using gradients and the learning rate
    a = a - lr * a_grad   # Line 27
    b = b - lr * b_grad   # Line 28
    
print(a, b)

# Sanity Check: do we get the same results as our gradient descent?
linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])
