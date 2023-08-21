import numpy as np

# from src.RNN import *
np.set_printoptions(linewidth=200)

def init_weights(dims, hidden_idx, seed):
    if seed is not None:
        np.random.seed(seed)

    weights = []

    for i in range(len(dims) - 1):
        if i in hidden_idx:
            reccurent_weights = np.random.randn(dims[i], dims[i])

            forward_weights = np.random.randn(dims[i] + 1, dims[i + 1])
            forward_weights[0, :] = np.random.randn(dims[i + 1]) * 0.01
            weights.append([forward_weights, reccurent_weights])

        else:
            forward_weights = np.random.randn(dims[i] + 1, dims[i + 1])
            forward_weights[0, :] = np.random.randn(dims[i + 1]) * 0.01
            weights.append(forward_weights)

    return weights


dims = [10, 32, 4]
hidden_idx = [1]
seed = 13337

weight_sets = init_weights(dims, hidden_idx, seed)
# print(len(weight_sets))
print(weight_sets[0][1][0])

weights = {
    "input_to_hidden": None,
    "hidden_to_hidden": None,
    "hidden_to_output": None,
}


