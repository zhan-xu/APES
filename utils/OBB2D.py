import numpy as np
from numpy.linalg import eigh, norm
from numpy import ndarray, array, asarray, dot, cross, cov, array, finfo


class OBB2D:
    def __init__(self, points=None):
        if points is not None:
            self.create_from_points(points)

    def create_from_points(self, points):
        covariance_matrix = np.cov(points, y=None, rowvar=0, bias=1)

        _, eigen_vectors = eigh(covariance_matrix)

        def try_to_normalize(v):
            n = norm(v)
            if n < finfo(float).resolution:
                raise ZeroDivisionError
            return v / n

        r = try_to_normalize(eigen_vectors[:, 0])
        u = try_to_normalize(eigen_vectors[:, 1])
        self.rotation = array((r, u))
        p_primes = asarray([self.rotation.dot(p) for p in points])

        self.min = np.min(p_primes, axis=0)
        self.max = np.max(p_primes, axis=0)


    def transform(self, point):
        return self.rotation.T.dot(array(point))


    @property
    def centroid(self):
        return self.transform((self.min + self.max) / 2.0)


    @property
    def extents(self):
        # return abs(self.transform((self.max - self.min) / 2.0))
        return abs((self.max - self.min) / 2.0 + 1.0)


    @property
    def anchors(self):
        return [
            self.centroid + self.rotation[0, :] * self.extents[0] + self.rotation[1, :] * self.extents[1],
            self.centroid + self.rotation[0, :] * self.extents[0] - self.rotation[1, :] * self.extents[1],
            self.centroid - self.rotation[0, :] * self.extents[0] + self.rotation[1, :] * self.extents[1],
            self.centroid - self.rotation[0, :] * self.extents[0] - self.rotation[1, :] * self.extents[1]
        ]

    def grids(self, interval=10):
        num_divide_i = max(int(self.extents[0] // interval), 2)
        num_divide_j = max(int(self.extents[1] // interval), 2)
        pos = []
        for i in range(num_divide_i):
            for j in range(num_divide_j):
                pos_new = self.centroid + (2.0 * i / (num_divide_i - 1) - 1.0) * self.rotation[0, :] * self.extents[0] \
                          + (2.0 * j / (num_divide_j - 1) - 1.0) * self.rotation[1, :] * self.extents[1]
                pos.append(pos_new)
        return np.array(pos)
