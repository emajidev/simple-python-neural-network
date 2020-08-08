import numpy as np

from data_prep import features, targets, features_test, targets_test
from colors import Color

def sigmoid(x):
    return 1/(1+np.exp(-x))


# hyperparameters
n_hidden = 2  # number of unity hide layer
n_trining_test = 1000  # number of interactions on training test
alpha = 0.05  # learning rate
ult_cost = None
m, k = features.shape  # number of training examples,number of data size
# initiataion of weight
in_hide = np.random.normal(scale=1/k**.5, size=(k, n_hidden))

hide_out = np.random.normal(scale=1/k**0.5, size=n_hidden)

# training

for e in range(n_trining_test):
    # gradients
    gradient_in_hide = np.zeros(in_hide.shape)
    gradient_hide_out = np.zeros(hide_out.shape)

    # interate on training set

    for x, y in zip(features.values, targets):
        # forward pss
        z = sigmoid(np.matmul(x, in_hide))
        y_ = sigmoid(np.matmul(hide_out, z))  # prediction

        # backward pass
        salida_error = (y - y_) * y_ * (1 - y_)

        hide_error = np.dot(salida_error, hide_out) * z * (1 - z)

        gradient_in_hide += hide_error * x[:, None]
        gradient_hide_out += salida_error * z

        # update weight
    in_hide += alpha * gradient_in_hide / m
    hide_out += alpha * gradient_hide_out / m

    if e % (n_trining_test/10) == 0:
        z = sigmoid(np.dot(features.values, in_hide))
        y_ = sigmoid(np.dot(z, hide_out))

        # funcion de costo
        cost = np.mean((y_-targets)**2)

        if ult_cost and ult_cost < cost:
            print("training:   ",Color.red ,"{:.3f}".format(cost), "ADVERENCIA")
        else:
            print(Color.white,"training:   ",Color.blue ,"{:.3f}".format(cost))

        ult_cost = cost

# presicion data test,"{:.3f}".format
z = sigmoid(np.dot(features_test, in_hide))
y_ = sigmoid(np.dot(z, hide_out))

predictions = y_ > 0.5
precision = np.mean(predictions == targets_test)
print (Color.green,"Prediction: " , Color.yellow ,"{:.3f}".format(precision))

