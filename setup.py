import setuptools

import learnML

setuptools.setup(
    include_package_data=True,
    name="learnML",
    version=learnML.__version__,
    description="LearnML - Machine Learning Algorithms",
    url="https://github.com/GopalSaraf/learnML",
    author="Gopal Saraf",
    author_email="mail@gopalsaraf.com",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
    ],
    long_description="This package contains the implementation of various machine learning algorithms "
    "from scratch using Python 3.6. The algorithms are implemented as classes and are tested on different datasets. "
    "The algorithms are provided for educational purposes only and are not intended for production use.",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
