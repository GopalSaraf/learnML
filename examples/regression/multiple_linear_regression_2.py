import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from learnML.preprocessing import OneHotEncoder, train_test_split
from learnML.regression import MultipleLinearRegression

data = pd.read_csv("data/50_startups.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values


print(X.dtype)
print(X)

X = OneHotEncoder(X, indexes=3).fit_transform()
print(X.dtype)
print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = MultipleLinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
np.set_printoptions(precision=2)
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)
