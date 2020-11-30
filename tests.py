import utils
import numpy as np

a = np.array([[0, 3, 5, 7, 9]])
b = np.array([[0, 5, 8, 3, 4]])

X = np.random.rand(5, 100)
# print(utils.mse_loss(a, b))
#
# th_gr, bias_gr = utils.calc_gradient_descent(a, b, X)
# print(th_gr)
# print(bias_gr)
# #print(utils.predict(X))
# print(utils.init_weights(X))
# print(utils.init_bias(X))

model = utils.LinearRegression()
model.fit(X, a)
print(model.LOSS_HISTORY)