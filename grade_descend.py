import numpy as np
import matplotlib.pyplot as plt
import math


X = [0.5, 2.2, 2]
y = [2, 2.5, 1.4]

# Taxa de apredizagem
alpha = 0.01

# Coeficientes
w0 = 0.1 # Angular
w1 = 0.1 # Linear

# Hipótese
def y_hat(x, w0, w1):
    return w0 + w1 * x


def plot_line(X, y, w0, w1):
    x_values = [i for i in range(int(min(X)) - 1, int(max(X)) + 2)]
    y_values = [y_hat(x, w0, w1) for x in x_values]
    plt.plot(x_values, y_values, 'r')
    plt.plot(X, y, 'bo')
    plt.show()

# plot_line(X, y, w0, w1)

# Cálculo do custo
# Mean Squared Error
def MSE(X, y, w0, w1):
    custo = 0
    m = float(len(X))
    for i in range(0, len(X)):
        custo += (y_hat(X[i], w0, w1) - y[i])**2
    
    return custo
    

# Custo atual
MSE(X, y, w0, w1)
# 3.1282999999999994


def gradient_descent_step(w0, w1, X, y, alpha):
    erro_w0 = 0
    erro_w1 = 0
    m = float(len(X))

    for i in range(0, len(X)):
        erro_w0 = y_hat(X[i], w0, w1) - y[i]
        erro_w1 = (y_hat(X[i], w0, w1) - y[i]) * X[i]
    
    new_w0 = w0 - alpha * (1 / m) * erro_w0
    new_w1 = w1 - alpha * (1 / m) * erro_w1
    
    return new_w0, new_w1

w0, w1 = gradient_descent_step(w0, w1, X, y, alpha)
# 0.10366666666666667 0.10733333333333334

# Gradiente descendente loop
epoch = 1000 # Quantidade de repetições

def gradiente_descent(w0, w1, X, y, alpha, epoch):
    custo = np.zeros(epoch)
    for i in range(epoch):
        w0, w1 = gradient_descent_step(w0, w1, X, y, alpha)
        custo [i] = MSE(X, y, w0, w1)
    
    return w0, w1, custo

w0, w1, custo = gradiente_descent(w0, w1, X, y, alpha, epoch)

# Gráfico do custo
fig, ax = plt.subplots()
ax.plot(np.arange(epoch), custo, 'r')
ax.set_xlabel('Iterações')
ax.set_ylabel('Custo')
ax.set_title('MSE vs. Epoch')
plt.show()

plot_line(X, y, w0, w1)
plt.show()
