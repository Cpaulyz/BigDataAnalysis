import matplotlib.pyplot as plt

from data import feature_train, target_train, feature_test, target_test


class KNNClassifier:
    def __init__(self, feature, labels):
        self.feature = feature
        self.labels = labels

    def get_distance(self, feature_line1, feature_line2):
        tmp = 0
        for i in range(len(feature_line1)):
            tmp += (feature_line1[i] - feature_line2[i]) ** 2
        return tmp ** 0.5

    def get_type(self, k, feature_line):
        dic = {}
        for index in range(len(self.feature)):
            dist = self.get_distance(self.feature[index], feature_line)
            dic[index] = dist
        # sort
        sort_dic = sorted(dic.items(), key=lambda x: x[1], reverse=False)
        # print(sort_dic)
        vote = {}
        for i in range(k):
            index = sort_dic[i][0]
            type = self.labels[index]
            if type not in vote.keys():
                vote[type] = 1
            else:
                vote[type] += 1
        vote_rank = sorted(vote.items(), key=lambda x: x[1], reverse=True)
        # print(vote_rank)
        return vote_rank[0][0]

    def predict(self, k, feature):
        res = []
        for feature_line in feature:
            res.append(self.get_type(k, feature_line))
        return res

    def score(self, k, feature, labels):
        predict_set = self.predict(k, feature)
        return len([index for index in range(len(labels)) if predict_set[index] == labels[index]]) / len(labels)


knn = KNNClassifier(feature_train, target_train)
print("训练集:", knn.score(5, feature_train, target_train))
print("测试集:", knn.score(5, feature_test, target_test))
target_test_predict = knn.predict(5, feature_test)
target_train_predict = knn.predict(5,feature_train)
comp = zip(target_test, target_test_predict)
print(list(comp))


plt.figure()
plt.subplot(221)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.title('测试集-真实结果')
plt.scatter(feature_test[:, 0], feature_test[:, 1], c=target_test, edgecolors='k', s=50)
plt.subplot(222)
plt.title('测试集-预测结果')
plt.scatter(feature_test[:, 0], feature_test[:, 1], c=target_test_predict, edgecolors='k', s=50)
plt.subplot(223)
plt.title('训练集-真实结果')
plt.scatter(feature_train[:, 0], feature_train[:, 1], c=target_train, edgecolors='k', s=50)
plt.subplot(224)
plt.title('训练集-预测结果')
plt.scatter(feature_train[:, 0], feature_train[:, 1], c=target_train_predict, edgecolors='k', s=50)
plt.show()
