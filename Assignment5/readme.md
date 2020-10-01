# 数据集处理

## 数据获取

使用sklearn的dataset获取数据

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_feature = iris['data']
iris_target = iris['target']
iris_target_name = iris['target_names']
```

## 数据划分

使用sklearn自带的函数将其分割为训练集和测试集

* 训练集和测试集比例为2：1

* 为方便比较不同方法的优劣，我们固定随机数种子为10

```python
feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33，random_state=10)
```

## 可视化

使用plt对数据进行可视化，数据集展示如下

```python
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
```

![image-20200930161247489](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200930161247489.png)

# 方法1 DecisionTree

## 类定义

为了构建决策树，需要先定义节点类

```python
class Node:
    def __init__(self, dimension, threshold, isLeaf, left, right, species):
        self.dimension = dimension  # 划分维度
        self.threshold = threshold  # 划分阈值
        self.isLeaf = isLeaf  # 是否是叶节点
        self.left = left  # 左支（叶节点时为None）
        self.right = right  # 右支（叶节点时为None）
        self.species = species  # 分类（如果是叶节点）

```

## 构建决策树

决策树部分，采用CART算法构建决策树，下面将**按照依赖关系自底向上**介绍结构化方法

### 基尼值

计算公式为

![image-20200930162351311](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200930162351311.png)

基尼值越小说明该数据集中不同类的数据越少

p<sub>v</sub>代表了v类数据在总类中的频率

代码实现如下

```python
def get_gini(label):
    """
    计算GINI值
    :param label: 数组，里面存的是分类
    :return: 返回Gini值
    """
    gini = 1
    dic = {}
    for target in label:
        if target in dic.keys():
            dic[target] += 1
        else:
            dic[target] = 1
    for value in dic.values():
        tmp = value / len(label)
        gini -= tmp * tmp
    return gini
```

### 基尼系数

计算公式如下

![image-20200930162520099](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200930162520099.png)

因为鸢尾花数据集的属性都是浮点数，为了**二分化**，我们需要寻找一个阈值，这里采用的方法是枚举所有的划分情况，因此需要做：

1. 排序给定维度下的属性
2. 选取相邻属性值的平均值作为候选阈值，并去重
3. 遍历所有可能的阈值，选取基尼系数最小的划分阈值，返回基尼系数和划分阈值

代码实现如下

```python
def get_gini_index_min(feature, label, dimension):
    """
    获取某个维度的最小GiniIndex
    :param feature: 所有属性list
    :param label: 标记list
    :param dimension: 维度(从0开始)
    :return: gini_index(最小GiniIndex)  threshold(对应阈值)
    """
    attr = feature[:, dimension]
    gini_index = 1
    threshold = 0
    attr_sort = sorted(attr)
    candicate_thre = []
    # 寻找候选阈值
    for i in range(len(attr_sort) - 1):
        tmp = (attr_sort[i] + attr_sort[i + 1]) / 2
        if tmp not in candicate_thre:
            candicate_thre.append(tmp)
    # 寻找最小GiniIndex
    for thre_tmp in candicate_thre:
        index_small_list = [index for index in range(len(feature)) if attr[index] < thre_tmp]
        label_small_tmp = label[index_small_list]
        index_large_list = [index for index in range(len(feature)) if attr[index] >= thre_tmp]
        label_large_tmp = label[index_large_list]
        gini_index_tmp = get_gini(label_small_tmp) * len(label_small_tmp) / len(attr) + get_gini(label_large_tmp) * len(
            label_large_tmp) / len(attr)
        if gini_index_tmp < gini_index:
            gini_index = gini_index_tmp
            threshold = thre_tmp
    print(gini_index, threshold)
    return gini_index, threshold
```

### 寻找划分维度

鸢尾花数据集有四个维度的数据，我们需要确定选取哪个维度的数据作为划分依据，因此，我们**依次计算各个维度下的最小基尼系数，选取最小基尼系数最小的维度作为划分维度**

有了上面计算最小基尼系数的方法，我们可以来选取基尼系数最小的数据维度

```python
def find_dimension_by_GiniIndex(feature, label):
    """
    寻找划分维度
    :param feature: 所有属性list
    :param label: 标记list
    :return: gini_index, threshold, dimension
    """
    dimension = 0
    threshold = 0
    gini_index_min = 1
    for d in range(len(feature[0])):
        gini_index, thre = get_gini_index_min(feature, label, d)
        if gini_index < gini_index_min:
            gini_index_min = gini_index
            dimension = d
            threshold = thre
    print(gini_index, threshold, dimension)
    return gini_index, threshold, dimension
```

### 构建决策树

有了以上的工具，就用递归的方法构建决策树了

递归的终点有两种情况

1. dataset只有一个元素了，那就不用再分了
2. dataset里有很多元素，但都是同一类型的，体现在GiniIndex=0，说明已经纯洁，不用再递归

实现如下

```python
def devide_by_dimension_and_thre(feature, label, threshold, dimension):
    """
    根据阈值和维度来划分数据集，返回小集和大集
    :param feature: 所有属性list
    :param label: 标记list
    :param threshold: 划分阈值
    :param dimension: 划分维度
    :return: feature_small, label_small, feature_large, label_large
    """
    attr = feature[:, dimension]
    index_small_list = [index for index in range(len(feature)) if attr[index] < threshold]
    feature_small = feature[index_small_list]
    label_small = label[index_small_list]
    index_large_list = [index for index in range(len(feature)) if attr[index] >= threshold]
    feature_large = feature[index_large_list]
    label_large = label[index_large_list]
    return feature_small, label_small, feature_large, label_large


def build_tree(feature, label):
    """
    递归构建决策树
    :param feature: 所有属性list
    :param label: 标记list
    :return: 决策树的根Node节点
    """
    if len(label) > 1:
        gini_index, threshold, dimension = find_dimension_by_GiniIndex(feature, label)
        if gini_index == 0:  # gini_index = 0，说明全都是同一种类型，就是叶节点
            return Node(dimension, threshold, True, None, None, label[0])
            print('end')
        else:
            # gini_index != 0，说明还不纯，继续划分，递归构建左支和右支
            feature_small, label_small, feature_large, label_large = devide_by_dimension_and_thre(feature, label,
                                                                                                  threshold,
                                                                                                  dimension)
            left = build_tree(feature_small, label_small)
            right = build_tree(feature_large, label_large)
            return Node(dimension, threshold, False, left, right, None)
    else:
        # 如果只有一个数据，直接是叶节点
        return Node(None, None, True, None, None, label[0])
```

## 可视化

使用graphviz对训练出的决策树进行可视化

![image-20200930172301085](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200930172301085.png)

## 分类结果

通过对测试集的预测来验证准确性

```python
def predict(root: Node, feature_line):
    """
    使用该方法进行预测
    :param root: 决策树根节点
    :param feature_line: 需要预测的属性值
    :return: 预测结构 label
    """
    node = root
    while not node.isLeaf:
        if feature_line[node.dimension] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node.species


def score(root, feature, label):
    """
    模型得分评估
    :param root: 决策树根节点
    :param feature: 测试集属性list
    :param label: 测试集标记list
    :return: 正确率
    """
    correct = 0
    for index in range(len(feature)):
        type = predict(root, feature[index])
        if type == label[index]:
            correct += 1
    print('correct rate is', correct / len(feature))
    
    
res = build_tree(feature_train, target_train)
score(res, feature_test, target_test)
```

得到正确率为0.96

经过验证，随机选取划分数据集的随机数种子（既按照2：1的训练集：测试集比例，随机划分），正确率都在90%以上，说明决策树方法能有效划分鸢尾花数据集

# 方法2 SVM

打算调库。。

# 方法3 BPNN

算法的实现总共分为6步：

1. 初始化参数
2. 前向传播
3. 计算代价函数
4. 反向传播
5. 更新参数
6. 模型评估

# 方法4 KNN

## KNN分类器实现

### 距离计算

计算公式为
$$
d=\sqrt{(x0-y0)^2+(x1-y1)^2+(x2-y2)^2+(x3-y3)^2}
$$

```python
    def get_distance(self, feature_line1, feature_line2):
        tmp = 0
        for i in range(len(feature_line1)):
            tmp += (feature_line1[i] - feature_line2[i]) ** 2
        return tmp ** 0.5
```

### 选择类型

直接选择距离最近的k-训练集中出现频率最高的种类作为分类结果

```python
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
```

### 完整代码

```python
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
```

## 分类结果

选取k=5，训练结果如下：

**训练集的准确率: 0.97**

**测试集的准确率: 0.96**

使用plt绘制分布图：

![image-20200930173159188](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200930173159188.png)

可以看出，KNN能够很好地对鸢尾花数据集进行分类