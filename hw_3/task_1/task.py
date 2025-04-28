import numpy as np

class Matrix:
    def __init__(self, data):
        self.data = np.array(data)
        self.rows, self.cols = self.data.shape

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError("Can only add Matrix to Matrix")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition")
        return Matrix(self.data + other.data)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrices must have the same dimensions for element-wise multiplication")
            return Matrix(self.data * other.data)
        elif isinstance(other, (int, float)):
            return Matrix(self.data * other)
        else:
            raise TypeError("Can only multiply Matrix by Matrix, int, or float")

    def __rmul__(self, other):
        return self.__mul__(other)
        

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise ValueError("Can only matrix multiply with another Matrix")
        if self.cols != other.rows:
            raise ValueError("Matrices must have compatible dimensions for matrix multiplication")
        return Matrix(self.data @ other.data)

    def __str__(self):
        return str(self.data)

    def save_to_file(self, filename):
        with open(filename, "w") as f:
            f.write(str(self))


np.random.seed(0)
matrix1 = Matrix(np.random.randint(0, 10, (10, 10)))
matrix2 = Matrix(np.random.randint(0, 10, (10, 10)))

matrix1.save_to_file('matrix_1.txt')
matrix2.save_to_file('matrix_2.txt')

matrix_sum = matrix1 + matrix2
matrix_sum.save_to_file("matrix+.txt")


matrix_mul = matrix1 * matrix2
matrix_mul.save_to_file("matrix_mul.txt")

matrix_matmul = matrix1 @ matrix2
matrix_matmul.save_to_file("matrix_matmul.txt") 


print("Operations completed and results saved to files.")