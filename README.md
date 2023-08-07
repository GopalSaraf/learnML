# LearnML - Machine Learning Algorithms

## `Under Development`

## Introduction

This repository contains the implementation of various machine learning algorithms from scratch using Python 3.6. The algorithms are implemented as classes and are tested on different datasets. The algorithms are provided for educational purposes only and are not intended for production use.

## Implemented Algorithms

1. Linear Regression:

- Univariate Linear Regression: `learnML.regression.univariate_regression.UnivariateLinearRegression`
- Multiple Linear Regression: `learnML.regression.multiple_regression.MultipleLinearRegression`

<br>

2. Feature Scaling

- Z-Score Normalization: `learnML.preprocessing.feature_scaling.ZScoreNormalization`

<br>

3. Feature Engineering

- Polynomial Features: `learnML.preprocessing.feature_engineering.PolynomialFeatures`

<br>

### Note

These classes are implemented using interfaces defined in the `learnML.interfaces` package. The interfaces are implemented within their respective classes and have the following methods:

1. `learnML.interfaces.imodel.IModel`:

   - `fit(X, y)`: Trains the model.
   - `predict(X)`: Predicts the output for the given input.

<br>

2. `learnML.interfaces.ifeature_scaling.IFeatureScaling`:

   - `fit_transform(X)`: Fits and transforms the input data.
   - `inverse_transform(X)`: Inverse transforms the input data.

<br>

3. `learnML.interfaces.ifeature_engineering.IFeatureEngineering`:
   - `transform(X)`: Transforms the input data.

<br>

## Installation

1. Clone the repository.
2. Install the dependencies using `pip install -r requirements.txt`.
3. Explore the examples in the `examples` folder.
4. Run the examples using `python <example_name>.py`.
5. Utilize the classes in your own code.
