import utils
import numpy as np

a = np.array([0, 3, 5, 7, 9])
b = np.array([0, 5, 8, 3, 4])

print(utils.mse_loss(a, b))
