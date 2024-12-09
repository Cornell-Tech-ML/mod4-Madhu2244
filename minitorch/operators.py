"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Compare two numbers."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if two numbers are equal."""
    return abs(x - y) < 1e-2


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """ReLU function."""
    return x if x > 0 else 0


def log(x: float) -> float:
    """Logarithm function."""
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function."""
    return math.exp(x)


def log_back(x: float, g: float) -> float:
    """Computes the derivative of log times a second arg."""
    return g / x


def inv(x: float) -> float:
    """Inverse function."""
    return 1.0 / x


def inv_back(x: float, g: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    return -g / (x**2)


def relu_back(x: float, g: float) -> float:
    """Computes the derivative of ReLU times a second arg."""
    return g if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[[float], float], xs: Iterable[float]) -> Iterable[float]:
    """Apply a function to each element of a list."""
    return [f(x) for x in xs]


def zipWith(
    f: Callable[[float, float], float], xs: Iterable[float], ys: Iterable[float]
) -> Iterable[float]:
    """Apply a function to two lists element-wise."""
    return [f(x, y) for x, y in zip(xs, ys)]


def reduce(
    f: Callable[[float, float], float], xs: Iterable[float], init: float
) -> float:
    """Reduce a list to a single value using a binary operation."""
    result = init
    for x in xs:
        result = f(result, x)
    return result


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate a list."""
    return map(neg, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add two lists."""
    return zipWith(add, xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum a list."""
    return reduce(add, xs, 0)


def prod(xs: Iterable[float]) -> float:
    """Multiply a list."""
    return reduce(mul, xs, 1)


# TODO: Implement for Task 0.3.
