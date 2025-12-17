import random
import numpy as np
import matplotlib.pyplot as plt

p = np.array([
    [1, -3, -4],
    [1, -2, 1],
    [1, 0, 1],
    [1, 2, 2],
    [1, -2, -4],
    [1, 0, -2],
    [1, 2, 1],
    [1, 3, -4]
], dtype=float)

t = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)

w = np.random.uniform(-1, 1, 3)
eta = 1.0
max_iter = 1000

def activation(u):
    return 1 if u >= 0 else -1

plt.ion()

for epoch in range(max_iter):
    idx = list(range(len(p)))
    random.shuffle(idx)
    errors = 0

    for i in idx:
        y = activation(np.dot(w, p[i]))
        if y != t[i]:
            w = w + eta * (t[i] - y) * p[i]
            errors += 1

    x1 = p[:, 1]
    x2 = p[:, 2]

    plt.clf()
    for i in range(len(p)):
        if t[i] == 1:
            plt.scatter(x1[i], x2[i], marker='o')
        else:
            plt.scatter(x1[i], x2[i], marker='x')

    xs = np.linspace(min(x1) - 1, max(x1) + 1, 100)
    if w[2] != 0:
        ys = -(w[0] + w[1] * xs) / w[2]
        plt.plot(xs, ys)

    plt.title(f"Iteracja {epoch + 1}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.pause(0.3)

    if errors == 0:
        break

plt.ioff()
plt.show()
