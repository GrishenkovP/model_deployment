import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

iris = datasets.load_iris()
iris_frame = pd.DataFrame(iris.data)
list_feature_names = [n_column.replace(" (cm)", "") \
                              .replace(" ", "_") for n_column in iris.feature_names]
iris_frame.columns = list_feature_names
iris_frame['target'] = iris.target
#iris_name = iris.target_names

X = iris_frame.iloc[:, iris_frame.columns != 'target']
y = iris_frame['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


model = KNeighborsClassifier()
model.fit(X_train.values, y_train)

filename = 'ml/model.pkl'
joblib.dump(model, filename)