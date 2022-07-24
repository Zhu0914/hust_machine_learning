from numpy import *


def sigmoid(x):
    return 1 / (1 + exp(-x))


def logistic_classify(test_mat, weight_arr):
    m = test_mat.shape[0]
    test_mat = append(test_mat, ones((m, 1)), axis=1)
    predic = sigmoid(dot(test_mat, weight_arr.T))
    predicted_vals = ones((m, 1))
    predicted_vals[predic < 0.5] = -1
    return predicted_vals


def training(data_mat, label_list, D, epochs=400, learn_rate=0.008, alpha=6, delta=0.03):
    label_list = label_list.reshape(-1, 1)
    m = data_mat.shape[0]
    data_mat = append(data_mat, ones((m, 1)), axis=1)
    n = data_mat.shape[1]
    weight_arr = ones((1, n)) / n
    for i in range(epochs):
        z = dot(data_mat, weight_arr.T)
        h_theta = sigmoid(z)
        grad = dot(data_mat.T, (h_theta - label_list) * D)
        gradNorm = linalg.norm(grad, ord=2)
        if gradNorm <= delta:
            print("satisfied")
            break

        else:
            weight_arr = weight_arr - learn_rate * grad.T

    predic = sigmoid(dot(data_mat, weight_arr.T))
    predictedVals = zeros((m, 1))
    predictedVals[predic > 0.5] = 1
    errArr = ones((m, 1))
    errArr[predictedVals == label_list] = 0
    weightedError = dot(D.reshape(1, -1), errArr)
    return weight_arr, weightedError, predictedVals


def train_logistic(data_mat, class_label, numIt=10):
    weakClassArr = []
    m = shape(data_mat)[0]
    D = ones((m, 1)) / m
    aggClassEst = zeros((m, 1))

    for i in range(numIt):
        bestWeights = {}
        weightArr, error, classEst = training(data_mat, class_label, D)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestWeights['alpha'] = alpha
        bestWeights['weights'] = weightArr
        weakClassArr.append(bestWeights)
        expon = alpha * ((class_label.reshape(-1, 1) != classEst) * 2 - 1).reshape(-1, 1)
        D = D * exp(expon)
        D = D / D.sum()
        aggClassEst += alpha * classEst
    return weakClassArr, aggClassEst


def classify_logistic(data_mat, classifier_arr):
    m = shape(data_mat)[0]
    aggClassEst = zeros((m, 1))
    # thresh_val = 0
    for i in range(len(classifier_arr)):
        aggClassEst += classifier_arr[i]['alpha'] * logistic_classify(data_mat,
                                                                      classifier_arr[i]['weights'].reshape(1, -1))
    returnMat = zeros((m, 1))
    returnMat[aggClassEst >= 0] = 1
    return returnMat
