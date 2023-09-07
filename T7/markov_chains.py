import numpy as np

p = np.array([[0.5, 0.5, 0, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0, 0],
    [0, 0.5, 0, 0.5, 0, 0],
    [0, 0, 0.5, 0, 0.5, 0],
    [0, 0, 0, 0.5, 0, 0.5],
    [0, 0, 0, 0, 0.5, 0.5]])

x0 = np.array([0, 0, 1, 0, 0, 0])
x1 = np.matmul(x0, p)

x2 = np.matmul(x0, np.linalg.matrix_power(p, 2))
x4 = np.matmul(x0, np.linalg.matrix_power(p, 4))
x10 = np.matmul(x0, np.linalg.matrix_power(p, 10))
x20 = np.matmul(x0, np.linalg.matrix_power(p, 20))


x100 = np.matmul(x0, np.linalg.matrix_power(p, 100))
#array([0.16666676, 0.16666657, 0.16666686, 0.16666648, 0.16666676, 0.16666657])