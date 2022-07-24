from numpy import *
import pandas as pd
import stump
import logistic


def kfold_split(data_mat, label_list, whole_index_list, not_chosen_list):
    selected_index_list = random.choice(not_chosen_list, size=int(floor(data_mat.shape[0] / 10)), replace=False)
    left_index_list = delete(whole_index_list, selected_index_list)
    not_chosen_list = list(set(not_chosen_list).difference(set(selected_index_list)))
    test_mat = data_mat[selected_index_list]
    test_label_list = label_list[selected_index_list]
    train_mat = data_mat[left_index_list]
    train_label_list = label_list[left_index_list]
    return train_mat, train_label_list, selected_index_list, test_mat, test_label_list, not_chosen_list


class Adaboost():
    classArr = None

    def __init__(self, base):
        '''
        :param base: 基分类器编号 0 代表对数几率回归 1 代表决策树桩
        在此函数中对模型相关参数进行初始化，如后续还需要使用参数请赋初始值
        '''
        self.base = base

    def fit(self, x_file, y_file):
        '''
        在此函数中训练模型
        :param x_file:训练数据(data.csv)
        :param y_file:训练数据标记(targets.csv)
        '''
        data_mat = loadtxt(x_file, float, delimiter=',')
        label_list = loadtxt(y_file, float, delimiter=',')

        if self.base == 0:  # 基分类器为决策树桩
            min_error = inf
            for numIt in [1, 5, 10, 100]:
                whole_index_list = array([i for i in range(data_mat.shape[0])])
                not_chosen_list = array([i for i in range(data_mat.shape[0])])
                for i in range(10):
                    output_file_name = "./experiments/base%d_fold%d.csv" % (numIt, i + 1)
                    train_mat, train_label_list, selected_index_list, test_mat, test_label_list, not_chosen_list = kfold_split(
                        data_mat, label_list, whole_index_list, not_chosen_list)
                    weak_class_arr, agg_class_est = stump.train_stump(train_mat, train_label_list, numIt)
                    ans_list = stump.classify_stump(test_mat, weak_class_arr)  # 预测分类结果
                    ans_mat = selected_index_list.reshape(-1, 1)
                    ans_mat = append(ans_mat, ans_list.reshape(-1, 1), axis=1)
                    savetxt(output_file_name, ans_mat, delimiter=',')

                    corr_list = zeros((ans_list.shape[0], 1))
                    corr_list[ans_list != test_label_list.reshape(-1, 1)] = 1
                    error_rate = float(corr_list.sum() / corr_list.shape[0])
                    if error_rate < min_error:
                        min_error = error_rate
                        self.classArr = weak_class_arr

        elif self.base == 1:  # 基分类器为对数几率回归
            m, n = data_mat.shape
            for i in range(n):
                mean = data_mat[:, i].sum() / m  # 第i个特征值的平均值
                st_deviation = std(data_mat[:, i])  # 第i个特征值的标准差
                data_mat[:, i] = (data_mat[:, i] - mean) / st_deviation  # 对该特征值进行零均值规范化
            min_error = inf
            for numIt in [1, 5, 10, 100]:
                whole_index_list = array([i for i in range(data_mat.shape[0])])
                not_chosen_list = array([i for i in range(data_mat.shape[0])])
                errorrate = 0
                for i in range(10):
                    output_file_name = "./experiments/base%d_fold%d.csv" % (numIt, i + 1)
                    train_mat, train_label_list, selected_index_list, test_mat, test_label_list, not_chosen_list = kfold_split(
                        data_mat, label_list, whole_index_list, not_chosen_list)
                    weak_class_arr, agg_class_est = logistic.train_logistic(train_mat, train_label_list,
                                                                            numIt)
                    ans_list = logistic.classify_logistic(test_mat, weak_class_arr)  # 预测的分类结果
                    ans_mat = selected_index_list.reshape(-1, 1)
                    ans_mat = append(ans_mat, ans_list.reshape(-1, 1), axis=1)
                    savetxt(output_file_name, ans_mat, delimiter=',')
                    corr_list = zeros((ans_list.shape[0], 1))
                    corr_list[ans_list != test_label_list.reshape(-1, 1)] = 1
                    error_rate = float(corr_list.sum() / corr_list.shape[0])
                    errorrate += error_rate / 10
                    if error_rate < min_error:
                        min_error = error_rate
                        self.classArr = weak_class_arr
                print(errorrate)
            print(1 - min_error)

    def predict(self, x_file):
        '''
        :param x_file:测试集文件夹(后缀为csv)
        :return: 训练模型对测试集的预测标记
        '''
        test_mat = loadtxt(x_file, float, delimiter=',')
        m, n = test_mat.shape
        for i in range(n):
            mean = test_mat[:, i].sum() / m  # 第i个特征值的平均值
            st_deviation = std(test_mat[:, i])  # 第i个特征值的标准差
            test_mat[:, i] = (test_mat[:, i] - mean) / st_deviation  # 对该特征值进行零均值规范化
        if self.base == 0:
            ans_mat = stump.classify_stump(test_mat, self.classArr)
        elif self.base == 1:
            ans_mat = logistic.classify_logistic(test_mat, self.classArr)
        return ans_mat


if __name__ == '__main__':  # 运行函数
    dataList = loadtxt("data.csv", float, delimiter=',')
    labelList = loadtxt("targets.csv", float, delimiter=',')
    dataMat = dataList
    test1 = Adaboost(base=0)
    test1.fit('data.csv', 'targets.csv')

