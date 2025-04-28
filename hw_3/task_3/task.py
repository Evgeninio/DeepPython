import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

class Matrix(NDArrayOperatorsMixin):
    _cache = {}

    def __init__(self, data):
        self.data = np.array(data)
        self.rows, self.cols = self.data.shape

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, (Matrix, np.ndarray)):
                return NotImplemented
        inputs = tuple(x.data if isinstance(x, Matrix) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.data if isinstance(x, Matrix) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            return tuple(type(self)(x) for x in result)
        elif method == 'at':
            return None
        else:
            return type(self)(result)

    def __array_function__(self, func, types, args, kwargs):
        if func not in np._ArrayFunctionDispatcher._HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, (Matrix, np.ndarray)) for t in types):
            return NotImplemented
        return func(*[x.data if isinstance(x, Matrix) else x for x in args], **kwargs)

    def __hash__(self):
        return int(np.sum(self.data)) % 1000000007

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = np.array(value)

    def __str__(self):
        return str(self.data)

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            np.savetxt(f, self.data, fmt="%s")

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError("Can only matrix multiply with another Matrix")
        if self.cols != other.rows:
            raise ValueError("Incompatible dimensions for matrix multiplication")

        hash1 = hash(self)
        hash2 = hash(other)
        cache_key = (hash1, hash2)

        if cache_key in self._cache:
            return self._cache[cache_key]

        result = Matrix(self.data @ other.data)
        self._cache[cache_key] = result
        return result

    def copy(self):
        return Matrix(self.data.copy())


def find_collision(size):
    matrices = []
    hashes = set()
    while True:
        matrix = Matrix(np.random.randint(0, 10, size))
        h = hash(matrix)
        if h in hashes:
            for m in matrices:
                if hash(m) == h and not np.array_equal(m.data, matrix.data):
                    return m, matrix
        matrices.append(matrix)
        hashes.add(h)


np.random.seed(1)
A = Matrix(np.random.randint(0, 10, (3, 3)))
B = Matrix(np.random.randint(0, 10, (3, 3)))

C, _ = find_collision((3, 3))
while not (hash(A) == hash(C) and not np.array_equal(A.data, C.data)):
    C, _ = find_collision((3, 3))

D = B.copy()

AB = A @ B
CD = C @ D

A.save_to_file("A.txt")
B.save_to_file("B.txt")
C.save_to_file("C.txt")
D.save_to_file("D.txt")
AB.save_to_file("AB.txt")
CD.save_to_file("CD.txt")

with open("hash.txt", "w") as f:
    f.write(f"Hash(A): {hash(A)}\n")
    f.write(f"Hash(C): {hash(C)}\n")
    f.write(f"Hash(AB): {hash(AB)}\n")
    f.write(f"Hash(CD): {hash(CD)}\n")

print("Matrices and results saved to files.")