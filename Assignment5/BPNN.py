import sys

import matplotlib.pyplot as plt
import numpy as np

from data import feature_train, target_train, feature_test, target_test

'''
@author: cyz
'''
test_features = feature_test
test_targets = target_test
train_features = feature_train
train_targets = target_train


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        :param input_nodes:  输入层节点个数
        :param hidden_nodes:  隐藏层节点个数
        :param output_nodes:  输出层节点个数
        :param learning_rate:  学习率
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                                         (self.output_nodes, self.hidden_nodes))

        self.lr = learning_rate  # 学习率
        self.activation_function = self.sigmoid

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def train(self, inputs_list, targets_list):
        # 正向传播
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs  # 因为的取值为0、1、2，所以这里不再用激活函数了，否则结果会被限制在0到1

        # 反向传播
        loss = 0.5 * (targets - final_outputs) ** 2  # 损失函数
        delta_output_out = final_outputs - targets
        delta_output_in = delta_output_out
        delta_weight_ho_out = np.dot(delta_output_in, hidden_outputs.T)

        delta_hidden_out = np.dot(self.weights_hidden_to_output.T, delta_output_in)
        delta_hidden_in = delta_hidden_out * hidden_outputs * (1 - hidden_outputs)
        delta_wih = np.dot(delta_hidden_in, inputs.T)

        self.weights_hidden_to_output -= (self.lr * delta_weight_ho_out)
        self.weights_input_to_hidden -= (self.lr * delta_wih)

        return loss

    def predict(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs.round()  # signals from final output layer

        return final_outputs

    def score(self, feature, labels):
        predict_set = self.predict(feature)[0]
        return len([index for index in range(len(labels)) if predict_set[index] == labels[index]]) / len(labels)


def MSE(y, Y):
    return np.mean((y - Y) ** 2)


epochs = 1000  # 训练次数
learning_rate = 0.001
hidden_nodes = 10
output_nodes = 1
batch_size = 50

input_nodes = train_features.shape[1]
network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
losses = {'train': []}

for e in range(epochs):  # 进行epochs次训练
    batch = np.random.choice(len(train_features), size=batch_size)  # 从训练集中随机挑选50个样本进行训练
    for record, target in zip(train_features[batch],
                              train_targets[batch]):
        network.train(record, target)

    train_loss = MSE(network.predict(train_features), train_targets)
    sys.stdout.write("\rProgress: " + str(100 * e / float(epochs))[:4]
                     + "% ... Training loss: " + str(train_loss)[:5])

    losses['train'].append(train_loss)
print('  训练完成')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.plot(losses['train'], label='Training loss')
plt.legend()
plt.ylim(ymax=0.5)
plt.title('损失函数')
# plt.show()

print("训练集:", network.score(feature_train, target_train))
print("测试集:", network.score(feature_test, target_test))
target_test_predict = network.predict(feature_test)[0]
target_train_predict = network.predict(feature_train)[0]
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
