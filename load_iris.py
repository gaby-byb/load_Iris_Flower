import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import linear_model, metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#convert iris into a dataframe
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df.info()

#what is missing?
print(df.isnull().sum())

#how is it divided?
df["target"].value_counts()

#which features show clear separation/overlapping?
sns.pairplot(df, hue="target", diag_kind="kde")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

#applying logistic regression
reg = linear_model.LogisticRegression(max_iter=10000, random_state=0)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print(f"Model accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")

print(confusion_matrix(y_test, y_pred))
#Logistic Regression Model

def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors": range(1,21),
        "metric": ["euclidean","manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_
best_model = tune_model(X_train, y_train)
print(best_model)

#KNN Model
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    matrix = confusion_matrix(y_test, prediction)

    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confusion matrix: ')
print(matrix)