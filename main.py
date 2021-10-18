from perceptron import Perceptron
from adaline import Adaline
from utils import bipolar, unipolar, prep_for_bipolar, prep_for_bias, predict
from random import uniform
import numpy as np

if __name__ == "__main__":
    data_set = [[np.array([0, 0]), 0], [np.array([0, 1]), 0], [np.array([1, 0]), 0], [np.array([1, 1]), 1]]
    test_set = [[np.array([0, 0.01]), 0], [np.array([0.01, 0.98]), 0], [np.array([0.98, 0.02]), 0],
                [np.array([0.99, 0.99]), 1]]
    theta = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alpha = [0.001, 0.005, 0.01, 0.05, 0.08, 1]
    mi = [0.001, 0.005, 0.01, 0.05, 0.08, 0.1]
    error = [0.09, 0.1, 0.11, 0.13, 0.15, 0.17, 0.2, 0.3, 0.4, 0.5]
    #w = np.array([0.1, 0.1, 0.1])
    my_range = (-0.1, 0.1)
    w = np.array([uniform(my_range[0], my_range[1]), uniform(my_range[0], my_range[1]), uniform(my_range[0], my_range[1])])
    epochs = 0
    perceptron = Perceptron(0.05, 0, w, unipolar)
    adaline = Adaline(0.005, w, 0.08)
    print(w)
    print(adaline.learn(data_set), predict(adaline, test_set, 0, unipolar))
    print(perceptron.learn(data_set), predict(perceptron, test_set, 0, unipolar))



    """
    data_set = [[np.array([-1, -1]), -1], [np.array([-1, 1]), -1], [np.array([1, -1]), -1], [np.array([1, 1]), 1]]
    
    data_set = [[np.array([0, 0]), 0], [np.array([0, 1]), 0], [np.array([1, 0]), 0], [np.array([1, 1]), 1]]
    w = np.array([0.01, 0.01])
    for a in alpha:
        perceptron = Perceptron(a, 0, w.copy(), unipolar)
        print(a, perceptron.learn(data_set))
    print(epochs)
        for e in error:
        adaline = Adaline(0.005, w, e)
        print(e, adaline.learn(data_set), adaline.predict(test_set, 0.4, bipolar))
    print(epochs)
    
        for _ in range(6):
        w = np.array([uniform(my_range[0], my_range[1]), uniform(my_range[0], my_range[1]), uniform(my_range[0], my_range[1])])
        perceptron_uni = Perceptron(0.05, 0, w.copy(), unipolar)
        perceptron_bi = Perceptron(0.05, 0, w.copy(), bipolar)
        print("Uni", perceptron_uni.learn(data_set))
        print("Bi", perceptron_bi.learn(data_set))
        
        for _ in range(6):
        w = np.array([uniform(my_range[0], my_range[1]), uniform(my_range[0], my_range[1]), uniform(my_range[0], my_range[1])])
        perceptron_uni = Perceptron(0.05, 0, w.copy(), unipolar)
        perceptron_bi = Perceptron(0.05, 0, w.copy(), bipolar)
        print(w,perceptron_bi.learn(data_set), perceptron_uni.learn(data_set))

    adaline = Adaline(mi[0], w, 0.55)

    print("Mi: {} Threshold: {} Epochs: {} Efficiency with theta {}: {}".format(adaline.mi,
                                                                                adaline.threshold,
                                                                                adaline.learn(data_set),
                                                                                0.5,
                                                                                adaline.predict(data_set, 0.5, bipolar))
          )

    adaline = Adaline(mi[0], w, 1.5)

    print("Mi: {} Threshold: {} Epochs: {} Efficiency with theta {}: {}".format(adaline.mi,
                                                                                adaline.threshold,
                                                                                adaline.learn(data_set),
                                                                                0.5,
                                                                                adaline.predict(data_set, 0.5, bipolar))
          )
          
        for _ in range(10):
        w = np.array([uniform(my_range[0], my_range[1]), uniform(my_range[0], my_range[1]), uniform(my_range[0], my_range[1])])
        adaline = Adaline(0.005, w, 0.15)
        epochs += adaline.learn(data_set)
    print(epochs)      
                                                                                
                                                                                """
