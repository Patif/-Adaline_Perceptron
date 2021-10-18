from utils import bipolar, prep_for_bipolar, prep_for_bias


class Perceptron:
    def __init__(self, alpha, theta, w, activation_function):
        self.__alpha = alpha
        self.__theta = theta
        self.__w = w
        self.__act_func = activation_function

    def learn(self, data_set):
        if self.__act_func == bipolar:
            data_set = prep_for_bipolar(data_set)
        if not self.__theta:
            data_set = prep_for_bias(data_set)
        misclassifications = None
        epochs = 0
        while misclassifications != 0:
            epochs += 1
            misclassifications = 0
            for x, d in data_set:
                z = (self.__w * x).sum()
                y = self.__act_func(z > self.__theta)
                delta = d - y
                misclassifications += abs(delta)
                self.__w += self.__alpha * delta * x
        return epochs



    @property
    def alpha(self):
        return self.__alpha

    @property
    def activation_function(self):
        return self.__act_func.__name__

    @property
    def theta(self):
        return self.__theta

    @property
    def w(self):
        return self.__w
