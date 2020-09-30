import matplotlib.pyplot as plt
from sklearn import svm

from data import feature_train, target_train, feature_test, target_test

svm_classifier = svm.SVC(C=1.0, kernel='rbf', decision_function_shape='ovr', gamma=0.01)
svm_classifier.fit(feature_train, target_train)

print("训练集:", svm_classifier.score(feature_train, target_train))
print("测试集:", svm_classifier.score(feature_test, target_test))
target_test_predict = svm_classifier.predict(feature_test)
comp = zip(target_test, target_test_predict)
print(list(comp))

plt.figure()
plt.subplot(121)
plt.scatter(feature_test[:, 0], feature_test[:, 1], c=target_test.reshape((-1)), edgecolors='k', s=50)
plt.subplot(122)
plt.scatter(feature_test[:, 0], feature_test[:, 1], c=target_test_predict.reshape((-1)), edgecolors='k', s=50)
plt.show()
