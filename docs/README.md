<h1 align="center">LearnML - Mastering Machine Learning Algorithms</h1>

<p align="center">
<img src="https://github.com/GopalSaraf/learnML/assets/83419951/5f8e7499-95bb-48e5-a198-eeb38e48b4a4" width="80%" height="auto" />
</p>

<br>

# Documentation for LearnML Algorithms

## Interfaces

- `IModel` : [ [Code](/learnML/interfaces/imodel.py) | [Documentation](/docs/interfaces/imodel.md) ] - Interface for all machine learning models.
- `IRegression` : [ [Code](/learnML/interfaces/iregression.py) | [Documentation](/docs/interfaces/iregression.md) ] - Interface for all regression models.
- `IFeatureEngineering` : [ [Code](/learnML/interfaces/ifeature_engineering.py) | [Documentation](/docs/interfaces/ifeature_engineering.md) ] - Interface for all feature engineering classes.

## Regression Models

- `UnivariateLinearRegression` : [ [Code](/learnML/regression/univariate_regression.py) | [Documentation](/docs/regression/univariate_regression.md) ] - Univariate linear regression model.
- `LinearRegression` : [ [Code](/learnML/regression/linear_regression.py) | [Documentation](/docs/regression/linear_regression.md) ] - Multivariate linear regression model.
- `PolynomialRegression` : [ [Code](/learnML/regression/polynomial_regression.py) | [Documentation](/docs/regression/polynomial_regression.md) ] - Polynomial regression model.

## Classification Models

- `LogisticRegression` : [ [Code](/learnML/classification/logistic_regression.py) | [Documentation](/docs/classification/logistic_regression.md) ] - Logistic regression model.
- `LinearSVC` : [ [Code](/learnML/classification/linear_svm.py) | [Documentation](/docs/classification/linear_svm.md) ] - Linear SVM model.

## Feature Engineering

- `ZScoreNormalization` : [ [Code](/learnML/preprocessing/z_score_normalization.py) | [Documentation](/docs/preprocessing/z_score_normalization.md) ] - Z-score normalization.
- `PolynomialFeatures` : [ [Code](/learnML/preprocessing/polynomial_features.py) | [Documentation](/docs/preprocessing/polynomial_features.md) ] - Polynomial feature generation.
- `OneHotEncoding` : [ [Code](/learnML/preprocessing/one_hot_encoding.py) | [Documentation](/docs/preprocessing/one_hot_encoding.md) ] - One-hot encoding.
- `train_test_split` : [ [Code](/learnML/preprocessing/train_test_split.py) | [Documentation](/docs/preprocessing/train_test_split.md) ] - Train-test split.

## Utilities

- `Metrics` : [ [Code](/learnML/utils/metrics.py) | [Documentation](/docs/utils/metrics.md) ] - Metrics for evaluating models.
