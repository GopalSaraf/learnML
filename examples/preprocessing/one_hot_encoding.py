import numpy as np

# Add the path of learnML to sys.path
import sys

sys.path.append("../../")

from learnML.preprocessing import OneHotEncoder

# Example usage:
data = np.array(
    [
        ["New York", "Male", 30],
        ["California", "Female", 25],
        ["Florida", "Male", 35],
        ["California", "Male", 28],
    ]
)

encoder = OneHotEncoder(data, indexes=0)
one_hot_encoded_data = encoder.fit()
encoder.transform()
# print("One-Hot Encoded Data:")
# print(one_hot_encoded_data)

# # Inverse Transform the data
# inverse_transformed_data = encoder.inverse_transform(one_hot_encoded_data)
# print("\nInverse Transformed Data:")
# print(inverse_transformed_data)
