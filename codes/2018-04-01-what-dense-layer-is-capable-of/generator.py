import numpy as np


class MatGenerator(object):
    def __init__(
            self, feature_shape, batch_size=32,
            shuffle_axes=False, seed=None):
        self.feature_shape = feature_shape
        self.feature_size = np.prod(feature_shape)
        self.shuffle_axes = shuffle_axes
        self.batch_size = batch_size

        self._rng = np.random.RandomState(seed=seed)
        self._perm = self._rng.permutation(self.feature_size)

    def get_batch(self):
        n_dim = len(self.feature_shape)
        data = np.zeros((self.batch_size, self.feature_size))
        labels = np.zeros([self.batch_size, n_dim])
        for i in range(self.batch_size):
            j = self._rng.randint(self.feature_size)
            data[i, j] = 1.0
            if self.shuffle_axes:
                j = self._perm[j]
            labels[i, ...] = np.unravel_index(j, self.feature_shape)
        return {'data': data, 'label': labels}

    def get_ref_batch(self):
        n_dim = len(self.feature_shape)
        data = np.zeros((self.feature_size, self.feature_size))
        labels = np.zeros([self.feature_size, n_dim])
        for i in range(self.feature_size):
            j = i
            data[i, j] = 1.0
            if self.shuffle_axes:
                j = self._perm[j]
            labels[i, ...] = np.unravel_index(j, self.feature_shape)
        return {'data': data, 'label': labels}
