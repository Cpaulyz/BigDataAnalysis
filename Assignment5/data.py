import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_feature = iris['data']
iris_target = iris['target']
iris_target_name = iris['target_names']

# 划分训练集和测试集
feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33,
                                                                          random_state=10)
# feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33,
#                                                                           random_state=10)

# print(iris_feature)
def show():
    t0 = [index for index in range(len(iris_target)) if iris_target[index] == 0]
    t1 = [index for index in range(len(iris_target)) if iris_target[index] == 1]
    t2 = [index for index in range(len(iris_target)) if iris_target[index] == 2]
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(x=iris_feature[t0, 0], y=iris_feature[t0, 1], color='r', label='Iris-virginica')
    plt.scatter(x=iris_feature[t1, 0], y=iris_feature[t1, 1], color='g', label='Iris-setosa')
    plt.scatter(x=iris_feature[t2, 0], y=iris_feature[t2, 1], color='b', label='Iris-versicolor')

    plt.xlabel("花萼长度")
    plt.ylabel("花瓣长度")
    plt.title("数据集展示")
    plt.show()
