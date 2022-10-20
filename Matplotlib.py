import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 4*np.pi, 0.1)
y = np.sin(x)

plt.plot(x,y)
plt.show()

x = np.arange(0, 4*np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel("x axis label")
plt.ylabel("y axis label")
plt.title("Sine and Cosine")
plt.legend(["Sine", "Cosine"])
plt.show()

# (row size, column size, index)
plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.title("Sine")

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title("Cosine")
plt.show()

x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
plt.scatter(x, y, s=None, c=None, marker=None, alpha=None)
plt.title("Scatter plot")
plt.show()

x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
plt.scatter(x, y, s=100, c='green', marker="^", alpha=0.5)
plt.title("Scatter plot")
plt.show()

X = np.arange(10)
Y = np.random.normal(0, 1, 10)
plt.bar(X, Y, )
plt.show()

data = np.random.normal(loc=0, scale=1, size=100)
plt.hist(data)
plt.show()

img = np.random.randint(0, 256, (128, 128))
plt.imshow(img)
plt.show()

img = np.random.randint(0, 256, (128, 128))
plt.imshow(img, cmap='gray')
plt.show()

img = np.random.randint(0, 256, (128, 128, 3))
plt.imshow(img)
plt.show()

img = np.random.randint(0, 256, (32, 32))
plt.imshow(img, cmap='coolwarm')
plt.show()
