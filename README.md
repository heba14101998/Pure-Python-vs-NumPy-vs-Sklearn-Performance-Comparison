

## Tools

* Python 3.
* Python basic modeules time, timeit, os, and itertools.
* NumPy Library (1.18.5).
* Matplotlib Library (3.3.2).
* Sklearn Library (1.0.1).
* Jupter Lab (3.2.5).
* Visual studio.

## Proplem Specification

It is technically possible to implement scalar and matrix calculations using `Python` lists. However, this can be unwieldy, and performance is poor when compared to languages suited for numerical computation, such as `MATLAB` or Fortran, or even some general purpose languages, such as `C` or `C++`.

To circumvent this deficiency, several libraries have emerged that maintain Python’s ease of use while lending the ability to perform numerical calculations in an efficient manner. Two such libraries worth mentioning are `NumPy` (one of the pioneer libraries to bring efficient numerical computation to Python) and `Sklrean` (Simple and efficient tools for predictive data analysisSimple and efficient tools for predictive data analysis).

Although it is possible to use this deterministic approach to estimate the coefficients of the linear model, it is not possible for some other models, such as neural networks. In these cases, iterative algorithms are used to 
estimate a solution for the parameters of the model.

One of the most-used algorithms is gradient descent, which at a high level consists of updating the parameter coefficients until we converge on a minimized loss (or cost). 

## Create Dataset with Gaussian noise

This program creates a set of 10,000 inputs x linearly distributed over the interval from 0 to 2. It then creates a set of desired outputs y = 3 + 2 * x + noise, where noise is taken from a Gaussian (normal) distribution with zero mean and standard deviation sigma = 0.1.

* X: A set of 10,000 inputs from 0 to 2.
* y = 3 + 2 * x + noise

**Note**: 
> The corrct values of the weights is 3 and 2.

## Model Using Pure Python

With a step size of 0.001 and 10,000 epochs, we can get a fairly precise estimate of w0 and w1. Inside the for-loop, the gradients with respect to the parameters are calculated and used in turn to update the weights, moving in the opposite direction in order to minimize the MSE cost function.

At each epoch, after the update, the output of the model is calculated. The vector operations are performed using **list comprehensions**. We could have also updated y in-place, but that would not have been beneficial to performance.

The elapsed time of the algorithm is measured using the time library. It takes $26.71$ seconds to estimate w0 = 2.9657 and w1 = 2.02859.


## Model Using NumPy Library
NumPy adds support for large multidimensional arrays and matrices along with a collection of mathematical functions to operate on them. I build a module called "MultipleLinearRegression" with the full implementation of linear regression algorithm using numpy.

The elapsed time of the algorithm is measured using the time library. It takes $0.9065$ seconds to estimate w0 = 2.9727 and w1 = 2.02263$. While the timeit library provide 0.88965 seconds.

## Model Using Scikit-lwarn library

The elapsed time of the algorithm is measured using the time library. It takes $0.0009975433$ seconds to estimate w0 = 3.0015 and w1 = 1.9982. While the timeit library provide 0.00034485 seconds.


<center style="font-size: 25px; background-color:lightskyblue;blue; font-family:Georgia">

# REFRENCES
</center>

* [Least squares approximation](https://en.wikipedia.org/wiki/Least_squares)
* [Linear regression from ML cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html)
* [Regression-metrics from machinelearningmastery](https://machinelearningmastery.com/regression-metrics-for-machine-learning/)
* [Linear regression a complete story](https://medium.com/analytics-vidhya/linear-regression-a-complete-story-c5edd37296c8)
* [Introduction to linear regression in python](https://towardsdatascience.com/introduction-to-linear-regression-in-python-c12a072bedf0)
* [Machine learning multivariate linear regression](https://medium.com/analytics-vidhya/machine-learning-multivariate-linear-regression-8f9878c0f56)
* [Numpy tensorflow performance from realpython](https://realpython.com/numpy-tensorflow-performance/https://realpython.com/numpy-tensorflow-performance/)
