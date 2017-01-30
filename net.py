import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import math


# np.random.seed(0)
x, y = sklearn.datasets.make_moons(200, noise=0.20)
x = np.array(x)
y = np.array(y)
y = y.reshape(len(y),1)

def sigmoid(val):
    return 1/(1+np.exp(-val))
def derivative(val):
    return val*(1-val)

def relu(val):
    return np.log(1+np.exp(val))


w0 = 2*np.random.random((2,10)) - 1
w1 = 2*np.random.random((10,1)) - 1

# batch train i times
for i in range(20000):
    l0 = x
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    l2_error = y - l2
    l2_delta = l2_error * derivative(l2)

    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * derivative(l1)

    l1_change = l0.T.dot(l1_delta)
    l2_change = l1.T.dot(l2_delta)
    w0 += l1_change
    w1 += l2_change
    if i%10000 ==0:
        print(i/10000)

print(w0)
print(w1)
def apply_model(coords):
    l1 = sigmoid(np.dot(coords, w0))
    l2 = sigmoid(np.dot(l1, w1))
    return l2


def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)

# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(x, y)

plot_decision_boundary(lambda x: apply_model(x))

plt.show()
