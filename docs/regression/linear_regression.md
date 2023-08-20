# Linear Regression

## Introduction

Linear regression is a supervised machine learning algorithm that is used to predict the value of a continuous variable. It is one of the simplest machine learning algorithms and is often used as a baseline for other algorithms.

Linear regression is used to model the relationship between a dependent variable and one or more independent variables. The dependent variable is the variable that we want to predict, and the independent variables are the variables that we use to predict the dependent variable.

The relationship between the dependent variable and the independent variables is assumed to be linear. This means that the dependent variable can be expressed as a linear combination of the independent variables.

The linear regression algorithm finds the best fit line that can be used to predict the value of the dependent variable for a given value of the independent variable.

## Implementation

The `LinearRegression` class is used to implement the linear regression algorithm. The class is defined in the [`linear_regression.py`](linear_regression.py) file.

The `LinearRegression` class has the following public methods:

- [`fit`](linear_regression.py#L15): Fit the linear regression model to the training data.
- [`predict`](linear_regression.py#L30): Predict the value of the dependent variable for the given independent variable.
- [`score`](linear_regression.py#L40): Calculate the coefficient of determination of the model.

The `LinearRegression` class has the following private methods:

- [`__init__`](linear_regression.py#L7): Initialize the class instance.
- [`__add_intercept`](linear_regression.py#L9): Add an intercept term to the independent variables.
- [`__gradient_descent`](linear_regression.py#L19): Perform gradient descent to find the optimal parameters.
- [`__cost_function`](linear_regression.py#L25): Calculate the cost function.
- [`__r2_score`](linear_regression.py#L35): Calculate the coefficient of determination.

## Usage

The `LinearRegression` class can be used to fit a linear regression model to a dataset. The following example demonstrates how to use the class:

```python
from learnml.linear_model import LinearRegression

# Create a LinearRegression object
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the value of the dependent variable for the test data
y_pred = model.predict(X_test)

# Calculate the coefficient of determination of the model
score = model.score(X_test, y_test)
```

## Example

The [`linear_regression.py`](../../examples/linear_regression.py) file contains an example that demonstrates how to use the `LinearRegression` class to fit a linear regression model to a dataset. The example uses the [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) to predict the median value of owner-occupied homes in Boston.

The dataset contains 506 rows and 14 columns. The columns are as follows:

- `CRIM`: Per capita crime rate by town
- `ZN`: Proportion of residential land zoned for lots over 25,000 sq. ft.
- `INDUS`: Proportion of non-retail business acres per town
- `CHAS`: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- `NOX`: Nitric oxides concentration (parts per 10 million)
- `RM`: Average number of rooms per dwelling
- `AGE`: Proportion of owner-occupied units built prior to 1940
- `DIS`: Weighted distances to five Boston employment centers
- `RAD`: Index of accessibility to radial highways
- `TAX`: Full-value property tax rate per $10,000
- `PTRATIO`: Pupil-teacher ratio by town

The target variable is `MEDV`, which is the median value of owner-occupied homes in $1000s.

The example uses the `LinearRegression` class to fit a linear regression model to the dataset. The model is then used to predict the median value of owner-occupied homes for the test data. The coefficient of determination of the model is also calculated.

The following is the output of the example:

```
Coefficients: [-0.10801136  0.04642042  0.02055863  2.68673382 -1.77957587  5.85775364
 -0.01190152 -0.96865216  0.17131143 -0.00939666 -0.3928038 ]
Intercept: 36.45948838509015
R2 score: 0.7334492147453086
```

The following plot shows the actual and predicted values of the target variable for the test data:

![Linear Regression](../../assets/images/linear_regression.png)

## References

- [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
- [Linear Regression for Machine Learning](https://machinelearningmastery.com/linear-regression-for-machine-learning/)
- [Linear Regression in Python](https://realpython.com/linear-regression-in-python/)
- [Linear Regression in Python using scikit-learn](https://stackabuse.com/linear-regression-in-python-using-scikit-learn/)
- [Linear Regression in Python - A Step-by-Step Guide](https://www.datacamp.com/community/tutorials/linear-regression-R)
- [Linear Regression in Python - Simple and Multiple Linear Regression](https://www.analyticsvidhya.com/blog/2021/05/linear-regression-in-python-simple-and-multiple-linear-regression/)
- [Linear Regression in Python - A Comprehensive Guide](https://www.analyticsvidhya.com/blog/2021/05/linear-regression-in-python-a-comprehensive-guide/)
- [Linear Regression in Python - A Complete Guide](https://www.analyticsvidhya.com/blog/2021/05/linear-regression-in-python-a-complete-guide/)
- [Linear Regression in Python - A Complete Guide](https://www.analyticsvidhya.com/blog/2021/05/linear-regression-in-python-a-complete-guide/)

## Project details

### Author

Seth M. Morton

### License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

### Acknowledgments

- [Linear Regression in Python](https://realpython.com/linear-regression-in-python/)
- [Linear Regression in Python using scikit-learn](https://stackabuse.com/linear-regression-in-python-using-scikit-learn/)
- [Linear Regression in Python - A Step-by-Step Guide](https://www.datacamp.com/community/tutorials/linear-regression-R)

### Related Blog Posts

- [Linear Regression in Python](https://sethmorton.com/2021/05/18/linear-regression-in-python/)

### Related Videos

- [Linear Regression in Python](https://youtu.be/2vJtbAha3To)
