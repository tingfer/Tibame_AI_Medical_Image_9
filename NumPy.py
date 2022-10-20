import numpy as np


def P13():
    print("\nP13 Create numpy.ndarray")
    x = np.arange(3)
    print(x)
    print(type(x))
    y = np.arange(5, dtype='float64')
    print(y)


def P14():
    print("\nP14 Create ndarray from Python List")
    python_list = [55, 66]
    numpy_array = np.array(python_list)
    print(python_list, type(python_list))
    print(numpy_array, type(numpy_array))


def P15():
    global x
    print("\nP15 Attributes of ndarray")
    x = np.arange(3)
    print(x)
    print(x.ndim)
    print(x.shape)
    print(x.size)
    print(x.dtype)
    x = np.array([[0, 1, 2], [3, 4, 5]])
    print(x)
    print(x.ndim)
    print(x.shape)
    print(x.size)


def P18():
    import numpy as np
    print("\nP18 Numpy Array Reshape")
    x = np.arange(6)
    print(x)
    new_shape = x.reshape(3, 2)
    print(new_shape)
    new_shape = np.reshape(x, (3, 2))
    print(new_shape)
    y = np.arange(6).reshape(2, 3)
    print(y)


def P19():
    print("\nP19 Initialize with Value")
    print(np.zeros(2))
    print(np.zeros((2, 3)))
    print(np.ones((2, 3)))
    print(np.full(shape=(2, 3), fill_value=5566))


def P20():
    print("\nP20 Initialize with Random Value")
    print(np.random.random(size=3))
    print(np.random.randint(low=1, high=10, size=(2, 2)))
    print(np.random.normal(loc=0.0, scale=1.0, size=(2, 2)))


def P21():
    global x
    print("\nP21 Array Index")
    x = np.arange(6)
    print(x[1], x[-3])
    x = np.arange(6).reshape(2, 3)
    print(x[0, 1], x[1, -2])


def P22():
    global x
    print("\nP22 Array Slice & Stride")
    x = np.arange(6)
    print(x[1:5])
    print(x[1:6])
    print(x[:3])
    print(x[1:4:2])
    x = np.arange(6).reshape(2, 3)
    print(x[0, 0:2])
    print(x[:, 1:])
    print(x[::1, ::2])
    print(x[:, ::-1])


def P23():
    global x
    print("\nP23 Boolean/Mask Index")
    x = np.arange(6)
    condition = x < 4
    print(x[condition])
    x[condition] = 0
    print(x)
    print(condition)


def P24():
    global x
    print("\nP24 Concatenate")
    x = np.array([[0, 1, 2], [3, 4, 5]])
    y = np.array([[6, 7, 8]])
    print(np.concatenate((x, y), axis=0))
    z = [[55], [66]]
    print(np.concatenate((x, z), axis=1))


def P25():
    print("\nP25 Basic Operations")
    a = np.array([[0, 1], [2, 3]])
    b = np.array([[4, 5], [6, 7]])
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)
    print(a - 3)
    print(a * 2)


def P26():
    global x
    print("\nP26 Basic Statistics")
    x = np.arange(10)
    print(np.min(x), x.min())
    print(np.max(x), x.max())
    print(np.mean(x), x.mean())
    print(np.std(x), x.std())
    print(np.argmax(x), x.argmax(), np.argmin(x), x.argmin())


def P27():
    global x
    print("\nP27 Expand_dim, squeeze")
    x = np.ones((128, 128))
    print(x.shape)
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    x = np.expand_dims(x, axis=-1)
    print(x.shape)
    x = np.squeeze(x, axis=-1)
    print(x.shape)


P13()
P14()
P15()
P18()
P19()
P20()
P21()
P22()
P23()
P24()
P25()
P26()
P27()

print("\nExercise Q1:")
print("Create a 1D array with values from 10 to 19 (Hint: np.arange)")
print(np.arange(10, 20))

print("\nExercise Q2:")
print("Revers a vector(first element becomes last)")
print(np.arange(10, 20)[::-1])

print("\nExercise Q3:")
print("Create a 3x3x3 array with random values (Hint: np.random.random or np.random.normal)")
print(np.random.random((3, 3, 3)))

print("\nExercise Q4:")
print("Create a 5x5 array with random values and find the minimum and maximum (Hint:np.min, np.max)")
arr = np.random.random((5, 5))
print(arr, arr.min(), arr.max())

print("\nExercise Q5:")
print("Add a border (filled with 0's) around an 3 x 3 matrix with width 1 by np.pad")
matrix = np.random.randint(low=1, high=10, size=(3, 3))
padMatrix = np.pad(matrix, pad_width=1, mode='constant', constant_values=0)
print(padMatrix)

print("\nExercise Q6:")
print("Normalize a 3x3 random matrix to 0~1 (Hint: (x - x min) / (x max - x min))")
x = np.random.randint(low=1, high=10, size=(3, 3))
min, max = x.min(), x.max()
print((x - min) / (max - min))

print("\nExercise Q7:")
print("Given a 1D array (0~12), negate all elements which are between 2 and 9")
arr = np.arange(13)
arr[(arr >= 2) & (arr <= 9)] *= -1
print(arr)

print("\nExercise Q8:")
print("Extract from the array [6, 7, 8, 10, 24, 45, 99,100] by the below conditions")
arr = np.array([6, 7, 8, 10, 24, 45, 99, 100])

print("(1) which are not divisible by 3 : ", arr[arr % 3 != 0])
print("(2) which are divisible by 5 : ", arr[arr % 5 == 0])
print("(3) which are divisible by 3 and 5 : ", arr[(arr % 3 == 0) & (arr % 5 == 0)])

print("\nExercise Q9:")
print("Create random vector of size 10 and replace the maximum value by 0 (np.argmax)")
vec = np.random.randint(low=1, high=100, size=10)
print(vec)
vec[vec.argmax()] = 0
print(vec)

print("\nExercise Q10:")
print("Create a 4x4 matrix and the row values are from 0 to 3")
print(np.zeros((4, 4)) + np.arange(4))
