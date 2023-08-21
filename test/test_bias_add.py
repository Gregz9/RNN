import numpy as np

np.set_printoptions(linewidth=200)

np.random.seed(1)

weights0 = np.random.randn(4, 32)
weights1 = np.random.randn(32, 64)

X = np.random.randn(10, 4)

z1 = X @ weights0

test_weights = np.random.randn(4,5) 
test_weights[0, :] = np.random.randn(test_weights.shape[1]) * 0.01
X = np.ones((5,4))
X[:, 0] = np.ones((X.shape[0])) * 0.01

print(X)
print(test_weights)

print(X@test_weights)


