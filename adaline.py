from utils import bipolar, prep_for_bipolar, prep_for_bias
from numpy import round


class Adaline:
    def __init__(self, mi, w, threshold_error):
        self.__mi = mi
        self.__w = w.copy()
        self.__threshold = threshold_error

    def learn(self, data_set):
        if len(data_set[0][0]) != len(self.__w):
            data_set = prep_for_bias(data_set)
        epochs = 0
        epsilon_s = None
        while not epsilon_s or epsilon_s > self.__threshold:
            epochs += 1
            epsilon_s = 0
            for x, d in data_set:
                delta = d - (self.__w * x).sum()
                epsilon_s += delta**2
                self.__w += round(2 * self.__mi * delta * x, 10)
            epsilon_s = epsilon_s/len(data_set)
        return epochs


    @property
    def mi(self):
        return self.__mi

    @property
    def threshold(self):
        return self.__threshold

    @property
    def w(self):
        return self.__w
