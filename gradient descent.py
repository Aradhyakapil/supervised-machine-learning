import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x,y):
  w_curr = b_curr = 0
  iterations = 1000
  n = len(x)
  learning_rate = 0.02
  w_values = []
  b_values = []
  for i in range(iterations):
    y_predicted = w_curr * x + b_curr
    cost = (1 / 2 * n) * sum([val ** 2 for val in (y_predicted-y)])
    wd = (1 / n) * sum(x * (y_predicted-y))
    bd = (1 / n) * sum(y_predicted-y)
    w_curr = w_curr - learning_rate * wd
    b_curr = b_curr - learning_rate * bd
    w_values.append(w_curr)
    b_values.append(b_curr)
    print ("m",w_curr, "b",b_curr, "cost",cost, "iteration",i)
  plt.scatter(x, y, label='Data Points')
  for w, b in zip(w_values, b_values):
    plt.plot(x, w * x + b, color='red', alpha=0.3)  # Plot the regression lines for each iteration
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()
  plt.show()
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
gradient_descent(x,y)
