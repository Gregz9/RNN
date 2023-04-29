for img in range(X.shape[3]):
    X_view = np.lib.stride_tricks.as_strided(X[..., img], shape=self.out_shape, strides=self.strides)
    for chout in range(self.feature_maps):
        conv = np.sum(self.kernel_tensor[:, chout, :, :, np.newaxis, np.newaxis] *
                      X_view[np.newaxis, ..., np.newaxis], axis=(2, 3))
        delta[:, :, chout, img] = np.sum(delta_next[..., chout, img] * np.rot90(np.rot90(conv)), axis=-1)
        kernel_grad[:, chout, ...] = np.einsum('ijkl,ijmn->klmn', X_view, delta_next[..., chout, img])


import numexpr as ne
from concurrent.futures import ThreadPoolExecutor

for img in range(X.shape[3]):
    X_view = np.lib.stride_tricks.as_strided(X[..., img], shape=self.out_shape, strides=self.strides)
    with ThreadPoolExecutor() as executor:
        futures = []
        for chout in range(self.feature_maps):
            conv = np.sum(self.kernel_tensor[:, chout, :, :, np.newaxis, np.newaxis] *
                          X_view[np.newaxis, ..., np.newaxis], axis=(2, 3))
            delta_view = delta[:, :, chout, img]
            delta_next_view = delta_next[..., chout, img]
            futures.append(executor.submit(ne.evaluate, 'sum(delta_next_view * rot90(rot90(conv)), axis=-1)', out=delta_view))
            for chin in range(self.input_channels):
                kernel_grad_view = kernel_grad[chin, chout, ...]
                X_pad_view = X_pad[..., chin, img]
                futures.append(executor.submit(ne.evaluate, 'sum(X_pad_view[x - start:x + end, y - start:y + end, :] * delta_next_view[x - start:x + end, y - start:y + end, np.newaxis], axis=(0, 1))', out=kernel_grad_view))
        for future in futures:
            future.result()

