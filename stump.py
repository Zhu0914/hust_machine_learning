from numpy import *


def stump_classify(data_mat, dimen, thresh_val,
                   thresh_ineq):
    m = shape(data_mat)[0]
    retArray = ones((m, 1))
    if thresh_ineq == 'lt':
        retArray[data_mat[:, dimen] <= thresh_val] = 0.0
    else:
        retArray[data_mat[:, dimen] > thresh_val] = 0.0
    return retArray


def build_stump(data_arr, class_labels, D):
    dataMat = data_arr
    labelMat = class_labels.reshape(-1, 1)
    m, n = shape(dataMat)
    num_stump = 10
    best_stump = {}
    best_class = zeros((m, 1))
    minError = inf
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        sortedValue = sort(dataMat[:, i])
        left = 0
        right = 299
        stepSize = (rangeMax - rangeMin) / num_stump
        while right < m:
            for inequal in ['lt', 'gt']:
                threshVal = (sortedValue[left] + sortedValue[right]) / 2
                predictedVals = stump_classify(dataMat, i, threshVal,
                                               inequal)
                err_arr = ones((m, 1))
                err_arr[predictedVals == labelMat] = 0
                weighted_error = dot(D.reshape(1, -1), err_arr)

                if weighted_error < minError:
                    minError = weighted_error
                    best_class = predictedVals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = threshVal
                    best_stump['ineq'] = inequal
            left += 300
            right += 300
    return best_stump, minError, best_class


def train_stump(data_mat, class_labels, num_it=10):
    weak_class_arr = []
    m = shape(data_mat)[0]
    D = ones((m, 1)) / m
    agg_class = zeros((m, 1))
    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_mat, class_labels, D)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        expon = alpha * ((class_labels.reshape(-1, 1) != class_est) * 2 - 1).reshape(-1, 1)
        D = D * exp(expon)
        D = D / D.sum()
        agg_class += alpha * class_est
    return weak_class_arr, agg_class


def classify_stump(dat_to_class, classifier_arr):
    dataMat = dat_to_class
    m = shape(dataMat)[0]
    agg_class = zeros((m, 1))
    for i in range(len(classifier_arr)):
        classEst = stump_classify(dataMat, classifier_arr[i]['dim'], classifier_arr[i]['thresh'],
                                  classifier_arr[i]['ineq'])
        agg_class += classifier_arr[i]['alpha'] * classEst
    return_mat = zeros((m, 1))
    minVal = agg_class.min()
    maxVal = agg_class.max()
    midVal = (minVal + maxVal) / 2
    return_mat[agg_class > midVal] = 1
    return return_mat
