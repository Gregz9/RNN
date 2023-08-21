import numpy as np
from typing import Union, Callable
from copy import deepcopy


class RecurrentNN:
    def __init__(
        self,
        dims: Union[tuple[int], list[int]],
        hidden_idx: Union[tuple[int], list[int]],
        h_function: Callable,
        o_function: Callable,
        cost_function: Callable,
        seed: int = None,
    ):
        self.dims = dims
        self.hidden_idx = hidden_idx
        self.h_function = h_function
        self.o_function = o_function
        self.cost_function = cost_function
        self.seed = seed

    def init_weights(self) -> None:
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = []

        for i in range(len(self.dims) - 1):
            if i in self.hidden_idx:
                reccurent_weights = np.random.randn(dims[i], dims[i])
                forward_weights = np.random.randn(dims[i] + 1, dims[i + 1])
                forward_weights[0, :] = np.random.randn(dims[i + 1]) * 0.01
                self.weights.append([forward_weights, reccurent_weights])

            else:
                forward_weights = np.random.randn(dims[i] + 1, dims[i + 1])
                forward_weights[0, :] = np.random.randn(dims[i + 1]) * 0.01
                self.weights.append(forward_weights)

    def _feedforward(self, X: np.ndarray) -> np.ndarray:
        # TODO: Store all activations/states when feeding the data forward

        self.z_matrices = []
        self.a_matrices = []
        self.h_matrices = []

        if len(X.shape) == 1:
            X.reshape((1, X.shape[0]))

        bias = np.ones((X.shape[0], 1)) * 0.01
        X = np.hstack([bias, X])

        self.a_matrices.append(X)
        self.z_matrices.append(X)
        a = X
        h = 0

        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                if len(self.wieghts) > 1:
                    z = a @ self.weights[i][0]
                    a = h @ self.weights[i][1] + z
                    h = self.h_function(a)

                    self.z_matrices.append(z)
                    self.h_matrices.append(h)

                    bias = np.ones((X.shape[0], 1)) * 0.01
                    a = np.hstack([bias, a])
                    self.a_matrices.append(a)

                else:
                    z = a @ self.weights[i] 
                    a = self.h_function(z) 
                    self.z_matrices.append(z)
                    self.a_matrices.append(a)

            else:
                


