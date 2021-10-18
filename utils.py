from copy import deepcopy
from numpy import insert


def bipolar(boolean):
    return 1 if boolean else -1


def unipolar(boolean):
    return 1 if boolean else 0


def prep_for_bipolar(data_set):
    data = deepcopy(data_set)
    for i, (values, label) in enumerate(data):
        data[i][1] = 2 * label - 1
        for j, val in enumerate(values):
            values[j] = 2 * val - 1
    return data


def prep_for_bias(data_set):
    data = deepcopy(data_set)
    for i, (values, _) in enumerate(data):
        data[i][0] = insert(values, 0, 1)
    return data


def predict(neuron, data_set, theta, act_function):
    if act_function == bipolar:
        data_set = prep_for_bipolar(data_set)
    if not theta:
        data_set = prep_for_bias(data_set)
    efficiency = 0
    for x, d in data_set:
        y = act_function((neuron.w * x).sum() > theta)
        efficiency += 1 if y == d else 0
    efficiency = efficiency / len(data_set)
    return efficiency
