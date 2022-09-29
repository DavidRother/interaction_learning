import matplotlib.pyplot as plt
import numpy as np

temperature = 1

q_values = np.asarray([12.2, 15.3, 10.2, 13.4, 14.3])
e = np.exp(q_values / temperature)
v = temperature * np.log(np.sum(e))
dist = np.exp((q_values - v) / temperature)
# print(dist)
dist = dist / np.sum(dist)

plt.plot(dist)

plt.show()
