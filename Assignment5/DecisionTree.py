from graphviz import Digraph

from data import feature_train, target_train, feature_test, target_test, iris_target_name


class Node:
    def __init__(self, dimension, threshold, isLeaf, left, right, species):
        self.dimension = dimension  # 划分维度
        self.threshold = threshold  # 划分阈值
        self.isLeaf = isLeaf  # 是否是叶节点
        self.left = left  # 左支（叶节点时为None）
        self.right = right  # 右支（叶节点时为None）
        self.species = species  # 分类（如果是叶节点）


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


def printtree(root: Node, indent='-', dict_tree={}, direct='L'):
    # 是否是叶节点
    if root.isLeaf:
        # print(root.species)
        dict_tree = {direct: str(root.species) + ' ' + iris_target_name[root.species]}

    else:
        left = printtree(root.left, indent=indent + "-", direct='L')
        left_copy = left.copy()
        right = printtree(root.right, indent=indent + "-", direct='R')
        right_copy = right.copy()
        left_copy.update(right_copy)
        stri = 'dimension:' + str(root.dimension) + "\nsplit:" + str(root.threshold) + "?"
        if indent != '-':
            dict_tree = {direct: {stri: left_copy}}
        else:
            dict_tree = {stri: left_copy}
    return dict_tree


def plot_model(tree, name):
    g = Digraph("G", filename=name, format='png', strict=False)
    first_label = list(tree.keys())[0]
    g.node("0", first_label)
    _sub_plot(g, tree, "0")
    g.view()


root = "0"


def _sub_plot(g, tree, inc):
    global root

    first_label = list(tree.keys())[0]
    ts = tree[first_label]
    for i in ts.keys():
        if isinstance(tree[first_label][i], dict):
            root = str(int(root) + 1)
            g.node(root, list(tree[first_label][i].keys())[0])
            g.edge(inc, root, str(i))
            _sub_plot(g, tree[first_label][i], root)
        else:
            root = str(int(root) + 1)
            g.node(root, tree[first_label][i])
            g.edge(inc, root, str(i))


res = build_tree(feature_train, target_train)
score(res, feature_test, target_test)
tree = printtree(res)
plot_model(tree, "hello.gv")

print(printtree(res))
