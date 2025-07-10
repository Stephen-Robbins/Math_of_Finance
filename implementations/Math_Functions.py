import math
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

def discrete_expected_value_and_variance(probabilities, values):
    """
    Calculates the expected value and variance of a discrete random variable.

    :param probabilities: list or array of probabilities [p_1, p_2, ..., p_n]
    :param values: list or array of corresponding values [x_1, x_2, ..., x_n]
    :return: (E, Var) where E is expected value and Var is variance
    """
    # Check that the lengths match
    if len(probabilities) != len(values):
        raise ValueError("The length of probabilities must match the length of values.")

    # Check that probabilities sum to 1 (within a small tolerance)
    if abs(sum(probabilities) - 1.0) > 1e-8:
        raise ValueError("The sum of the probabilities must be 1.")

    # Compute E(X)
    expected_value = sum(p * x for p, x in zip(probabilities, values))

    # Compute E(X^2)
    expected_value_sq = sum(p * (x**2) for p, x in zip(probabilities, values))

    # Variance = E(X^2) - [E(X)]^2
    variance = expected_value_sq - (expected_value**2)

    return expected_value, variance

def continuous_expected_value_and_variance(pdf_func, a, b):
    """
    Calculates the expected value and variance of a continuous random variable
    defined on [a, b] using numerical integration.

    :param pdf_func: A Python function that returns the PDF value at x, e.g., f(x).
    :param a: Lower bound of the domain.
    :param b: Upper bound of the domain.
    :return: (E, Var) tuple where
             E   = the expected value (float)
             Var = the variance (float)
    """

    # 1) Check if PDF integrates to 1 on [a, b]
    total_prob, _ = quad(pdf_func, a, b)
    if abs(total_prob - 1.0) > 1e-7:
        raise ValueError(f"PDF does not integrate to 1 over [{a}, {b}] (got {total_prob}).")

    # 2) Compute E(X) = ∫ x f(x) dx
    def integrand_for_mean(x):
        return x * pdf_func(x)

    E, _ = quad(integrand_for_mean, a, b)

    # 3) Compute E(X^2) = ∫ x^2 f(x) dx
    def integrand_for_mean_sq(x):
        return (x**2) * pdf_func(x)

    E_sq, _ = quad(integrand_for_mean_sq, a, b)

    # 4) Compute Variance = E(X^2) - [E(X)]^2
    variance = E_sq - (E**2)

    return E, variance

def normal_prob(b, mu, sigma_sq):
    """
    Calculate P(X <= b) where X ~ N(mu, sigma_sq).

    :param b: Upper limit (scalar).
    :param mu: Mean of the normal distribution (scalar).
    :param sigma_sq: Variance of the normal distribution (scalar).
    :return: Probability P(X <= b).
    """
    if sigma_sq <= 0:
        raise ValueError("Variance sigma_sq must be positive.")

    # Standard deviation
    sigma = math.sqrt(sigma_sq)

    # Standardize and use the standard normal CDF
    z = (b - mu) / sigma
    p = norm.cdf(z)
    return p

def clt(mu, sigma_sq, n):
    """
    Demonstrate the Central Limit Theorem for i.i.d. random variables X_i ~ (mu, sigma_sq).

    - S_n = X_1 + ... + X_n
      E[S_n] = n * mu
      Var(S_n) = n * sigma_sq

    - S_n / n
      E[S_n / n] = mu
      Var(S_n / n) = sigma_sq / n


    :param mu:       Mean of each X_i.
    :param sigma_sq: Variance of each X_i.
    :param n:        Number of i.i.d. trials.
    """

    if sigma_sq <= 0:
        raise ValueError("Variance must be positive.")
    if n <= 0:
        raise ValueError("Number of trials n must be positive.")

    # Mean and variance of S_n
    mean_Sn = n * mu
    var_Sn = n * sigma_sq

    # Mean and variance of S_n / n
    mean_SnOverN = mu
    var_SnOverN  = sigma_sq / n

    return mean_Sn, var_Sn, mean_SnOverN, var_SnOverN
