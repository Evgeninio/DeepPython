import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

class MyArray(NDArrayOperatorsMixin):
    def __init__(self, data):
        self._data = np.array(data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, (MyArray, np.ndarray)):
                return NotImplemented
        inputs = tuple(x._data if isinstance(x, MyArray) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x._data if isinstance(x, MyArray) else x
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
        if not all(issubclass(t, (MyArray, np.ndarray)) for t in types):
            return NotImplemented
        return func(*[x._data if isinstance(x, MyArray) else x for x in args], **kwargs)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = np.array(value)

    def __str__(self):
        return str(self._data)

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            np.savetxt(f, self._data, fmt="%s")

np.random.seed(0)
arr1 = MyArray(np.random.randint(0, 10, (10, 10)))
arr2 = MyArray(np.random.randint(0, 10, (10, 10)))

arr1.save_to_file("arr1.txt")
arr2.save_to_file("arr2.txt")

(arr1 + arr2).save_to_file("myarray+.txt")
(arr1 * arr2).save_to_file("myarray_mul.txt")
(arr1 @ arr2).save_to_file("myarray_matmul.txt")



print("Operations completed and saved to files.")