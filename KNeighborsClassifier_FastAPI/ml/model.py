import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import joblib

# **********************************************************************************************************************
iris = datasets.load_iris()
iris_frame = pd.DataFrame(iris.data)
iris_frame.columns = iris.feature_names
iris_frame['target'] = iris.target
iris_name = iris.target_names
print(iris_frame.head(10))
iris_frame.to_csv('iris_fame.csv', index=False)
# **********************************************************************************************************************
X = iris_frame.iloc[:, iris_frame.columns != 'target']
y = iris_frame['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = KNeighborsClassifier()
model.fit(X_train, y_train)
expected = y_test
predicted = model.predict(X_test)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)
err_train = np.mean(y_train != y_train_predict)
err_test = np.mean(y_test != y_test_predict)
print(round(err_train, 2), round(err_test, 2))
accuracy = accuracy_score(y_test, y_test_predict)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

filename = 'KNeighborsClassifier_model.pkl'
joblib.dump(model, filename)
# # **********************************************************************************************************************
# test_data = [5.9, 3.0 , 5.1, 1.8]
# pred_array = np.array(test_data)
# print(pred_array)
# preds = pred_array.reshape(1, -1)
# print(preds)
#
# model_open = open('ml/KNeighborsClassifier_model.pkl', 'rb')
# KNeighborsClassifier_model = joblib.load(model_open)
# model_prediction = KNeighborsClassifier_model.predict(preds)
# print(model_prediction[0])
