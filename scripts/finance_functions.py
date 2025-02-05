import numpy as np
from scipy.stats import norm

def black_scholes_call_price(S0, K, r, T, sigma):
    """
    Calculate the Black-Scholes price of a European call option (no dividend).

    Parameters:
    - S0: float, Current stock price.
    - K: float, Strike price.
    - r: float, Risk-free interest rate (annualized).
    - T: float, Time to maturity (in years).
    - sigma: float, Volatility of the underlying asset (annualized).

    Returns:
    - call_price: float, The Black-Scholes price of the call option.
    """
    # Compute d1 and d2 using the Black-Scholes formulas.
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate the call price.
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price