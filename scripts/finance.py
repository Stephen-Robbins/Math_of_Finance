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


def estimate_annualized_volatility(returns, trading_days=252):
    """
    Estimate the annualized volatility of a stock based on its daily returns.
    
    The function expects a list or NumPy array of daily returns (as decimals),
    and computes the log returns as:
        log_return_n = ln(1 + R_n)
    where R_n is the daily return:
        R_n = (S(n) - S(n-1)) / S(n-1)
    
    It then estimates the daily mean and daily volatility (standard deviation)
    as:
        μ_d = (1/N) ∑ ln(1 + R_n)
        σ_d = sqrt( (1/(N-1)) * ∑ (ln(1 + R_n) - μ_d)^2 )
    
    Finally, it annualizes these estimates using:
        μ = 252 * μ_d
        σ = σ_d * sqrt(252)
    
    Parameters:
    - returns: list or np.array of daily returns (in decimal form, e.g., 0.01 for 1%)
    - trading_days: number of trading days per year (default is 252)
    
    Returns:
    - annualized_volatility: The annualized volatility (σ) as a decimal.
    - daily_volatility: The computed daily volatility (σ_d) as a decimal.
    - daily_mean: The computed daily mean log return (μ_d).
    """
    # Convert the input to a NumPy array in case it's a list.
    returns = np.array(returns)
    
    # Compute log returns: ln(1 + R_n)
    log_returns = np.log(1 + returns)
    
    # Number of observations
    N = len(log_returns)
    if N < 2:
        raise ValueError("At least two returns are required to compute volatility.")
    
    # Calculate the daily mean log return
    daily_mean = np.mean(log_returns)
    
    # Calculate the daily volatility (sample standard deviation with N-1 in the denominator)
    daily_volatility = np.std(log_returns, ddof=1)
    
    # Annualize the daily volatility
    annualized_volatility = daily_volatility * np.sqrt(trading_days)
    
    return annualized_volatility, daily_volatility, daily_mean