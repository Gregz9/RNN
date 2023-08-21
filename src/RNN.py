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

    def _feedforward(self, X_batch): -> np.ndarray: 
    




