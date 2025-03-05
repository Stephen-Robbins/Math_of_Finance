import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime
from scipy.optimize import newton
import pandas as pd  
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import yfinance as yf
from datetime import datetime, timedelta


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


def annualized_volatility(returns, trading_days=252):
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

def historical_annualized_volatility(ticker, start, end, trading_days=252):
    """
    Download historical data for the given ticker between start and end dates,
    compute the daily returns, and then estimate the annualized volatility.
    
    Parameters:
    - ticker: str, stock ticker symbol (e.g., 'AAPL')
    - start: str, start date in 'YYYY-MM-DD' format
    - end: str, end date in 'YYYY-MM-DD' format
    - trading_days: int, number of trading days per year (default=252)
    
    Returns:
    - annualized_volatility: The annualized volatility (σ) as a decimal.
    - daily_volatility: The computed daily volatility (σ_d) as a decimal.
    - daily_mean: The computed daily mean log return (μ_d).
    """
    # Download historical data using yfinance
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker} between {start} and {end}.")
    
    # Compute daily returns based on closing prices
    # The percentage change gives R_n = (S(n) - S(n-1)) / S(n-1)
    daily_returns = data['Close'].pct_change().dropna()
    
    # Use the previously defined function to estimate volatility
    ann_vol, daily_vol, daily_mu = annualized_volatility(daily_returns.values, trading_days)
    
    return ann_vol, daily_vol, daily_mu

def get_current_stock_price(ticker):
    """
    Get the current (real-time) stock price for the given ticker.
    
    Parameters:
    - ticker: str, the stock ticker symbol (e.g., 'AAPL')
    
    Returns:
    - price: float, the current market price of the stock.
    
    Note: This uses yfinance's info attribute which provides the regular market price.
    """
    try:
        stock = yf.Ticker(ticker)
        # Option 1: Using fast_info (if available)
        if hasattr(stock, 'fast_info'):
            price = stock.fast_info['lastPrice']
        else:
            # Fallback: use the info dictionary
            price = stock.info.get('regularMarketPrice')
        if price is None:
            raise ValueError("Current price not available.")
        return price
    except Exception as e:
        print(f"Error retrieving current stock price for {ticker}: {e}")
        return None
import yfinance as yf

def get_current_option_price(ticker, expiration, strike, option_type='call'):
    """
    Get the current market price for an option on the given stock.
    
    Parameters:
    - ticker: str, the stock ticker symbol (e.g., 'AAPL')
    - expiration: str, the expiration date in 'YYYY-MM-DD' format
    - strike: float, the option's strike price
    - option_type: str, 'call' or 'put' (default is 'call')
    
    Returns:
    - option_price: float, the current price (last trade price) of the option.
    
    Raises a ValueError if the option is not found.
    """
    try:
        stock = yf.Ticker(ticker)
        # Get the option chain for the specified expiration date.
        opt_chain = stock.option_chain(expiration)
        
        # Select the appropriate DataFrame based on option type.
        if option_type.lower() == 'call':
            df = opt_chain.calls
        elif option_type.lower() == 'put':
            df = opt_chain.puts
        else:
            raise ValueError("option_type must be either 'call' or 'put'.")
        
        # Find the row(s) where the strike price matches (or nearly matches) the input.
        # Here we assume an exact match. (You could add tolerance if needed.)
        option_row = df[df['strike'] == strike]
        if option_row.empty:
            raise ValueError(f"No {option_type} option found with strike {strike} for {ticker} expiring on {expiration}.")
        
        # Return the last traded price for the option.
        # Note: Field name might be 'lastPrice' or similar.
        option_price = option_row.iloc[0]['lastPrice']
        return option_price
    except Exception as e:
        print(f"Error retrieving option price for {ticker}: {e}")
        return None
    
def time_to_maturity(expiration_date, current_date=None):
    """
    Calculate time to maturity (T) in years from the current date to the expiration date.
    
    Parameters:
    - expiration_date: str, expiration date in "YYYY-MM-DD" format (e.g., "2026-01-16")
    - current_date: datetime, optional, current date; if None, defaults to today's date.
    
    Returns:
    - T: float, time to maturity in years.
    
    Note:
    - This function uses 365.25 days per year to account for leap years.
    """
    if current_date is None:
        current_date = datetime.today()
        
    # Parse the expiration date
    expiration = datetime.strptime(expiration_date, "%Y-%m-%d")
    
    if expiration < current_date:
        raise ValueError("Expiration date is in the past.")
    
    # Compute the difference in days
    delta = expiration - current_date
    days = delta.days
    
    # Convert days to years
    T = days / 365.25
    return T



def black_scholes_price(option_type, S0, K, r, T, sigma):
    """
    Calculate the Black-Scholes price of a European call or put option (no dividends).

    Parameters:
    - option_type: str, 'call' or 'put'
    - S0: float or array-like, Current stock price.
    - K: float, Strike price.
    - r: float, Risk-free interest rate (annualized).
    - T: float, Time to maturity (in years).
    - sigma: float, Volatility of the underlying asset (annualized).

    Returns:
    - price: float or array-like, The Black-Scholes price of the option.
    """
    # Compute d1 and d2 (works with scalar or array-like S0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Please choose 'call' or 'put'.")
    
    return price

def monte_carlo_option_price(S0, E, r, T, sigma, payoff_func=None, n_sims=10000):
    """
    Estimate the price of an option using Monte Carlo simulation with a user-specified payoff.

    The terminal stock price S_T is simulated under the risk-neutral measure:
    
        S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),
    
    where Z ~ N(0,1). The payoff is then calculated for each simulation, and its
    discounted average is returned as the Monte Carlo price.

    Parameters:
    - S0: float, current stock price.
    - E: float, strike price (or any parameter needed by the payoff function).
    - r: float, risk-free interest rate (annualized).
    - T: float, time to maturity (in years).
    - sigma: float, volatility of the underlying asset (annualized).
    - payoff_func: function or None. A function that accepts an array of simulated 
          terminal prices S_T and returns an array of payoffs. If None, defaults to the 
          European call payoff: lambda S_T: np.maximum(S_T - E, 0).
    - n_sims: int, number of Monte Carlo simulations (default: 10,000).

    Returns:
    - mc_price: float, the Monte Carlo estimated option price.
    """
    # Default to the European call option payoff if no function is provided.
    if payoff_func is None:
        payoff_func = lambda S_T: np.maximum(S_T - E, 0)
    
    # Generate n_sims random samples from the standard normal distribution.
    Z = np.random.standard_normal(n_sims)
    
    # Simulate the terminal stock prices.
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate the payoff using the provided function.
    payoffs = payoff_func(S_T)
    
    # Discount the average payoff to obtain the present value.
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    return mc_price

def gbm_lognormal_stats(x, X_t, mu, sigma, T_minus_t):
    """
    Calculate various statistics for a lognormal variable arising from a 
    geometric Brownian motion (GBM) process at a future time.
    
    Parameters:
        x (float or array-like): The value(s) at which to evaluate the CDF and PDF.
        X_t (float): The value of the process at time t.
        mu (float): Drift coefficient of the GBM process.
        sigma (float): Volatility coefficient of the GBM process.
        T_minus_t (float): Time difference between the future time T and current time t.
        
    Returns:
        dict: A dictionary containing:
            'cdf'      : Cumulative Distribution Function evaluated at x.
            'pdf'      : Probability Density Function evaluated at x.
            'mean'     : Mean of the lognormal distribution.
            'median'   : Median of the lognormal distribution.
            'mode'     : Mode of the lognormal distribution.
            'variance' : Variance of the lognormal distribution.
    """
    # Update the parameters for the lognormal distribution
    mu_1 = np.log(X_t) + (mu - 0.5 * sigma**2) * T_minus_t
    sigma_1 = sigma * np.sqrt(T_minus_t)
    
    # Calculate the CDF and PDF at x
    # Note: For x > 0; x can be a scalar or an array.
    cdf = norm.cdf(np.log(x), loc=mu_1, scale=sigma_1)
    pdf = norm.pdf(np.log(x), loc=mu_1, scale=sigma_1) / x  # Adjusted for lognormal
    
    # Compute moments of the lognormal distribution
    mean = np.exp(mu_1 + 0.5 * sigma_1**2)
    median = np.exp(mu_1)
    mode = np.exp(mu_1 - sigma_1**2)
    variance = (np.exp(sigma_1**2) - 1) * np.exp(2 * mu_1 + sigma_1**2)
    
    stats= {
        'cdf': cdf,
        'pdf': pdf,
        'mean': mean,
        'median': median,
        'mode': mode,
        'variance': variance
    }
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value:.4f}" if np.isscalar(value) else f"{key.capitalize()}: {value}")


class Option:
    def __init__(self, S, E, T, r, sigma=None, premium=None, option_type='call', q=0.0):
        """
        Initialize an Option instance.
        
        Parameters:
            S : float
                Current underlying asset price.
            E : float
                Strike price.
            T : float
                Time to expiration (in years).
            r : float
                Annual risk-free interest rate.
            sigma : float, optional
                Volatility (if known). If not provided, premium must be given.
            premium : float, optional
                Market price of the option (used to back out implied volatility if sigma is not provided).
            option_type : str, default 'call'
                Type of option: 'call' or 'put'.
            q : float, default 0.0
                Continuous dividend rate.
        """
        self.S = S
        self.K = E
        self.T = T
        self.r = r
        self.q = q  # Added continuous dividend rate
        self.option_type = option_type.lower()
        
        if sigma is None and premium is None:
            raise ValueError("Provide either volatility (sigma) or premium (market price).")
        
        # If sigma is not provided, compute implied volatility from the market price.
        if sigma is None:
            self.premium = premium
            self.sigma = self.implied_volatility(premium)
        else:
            self.sigma = sigma
            self.premium = premium if premium is not None else self.price()
    
    def d1(self, sigma=None):
        """
        Calculate the d1 term used in the Black-Scholes formulas.
        """
        sigma = sigma if sigma is not None else self.sigma
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T))
    
    def d2(self, sigma=None):
        """
        Calculate the d2 term used in the Black-Scholes formulas.
        """
        sigma = sigma if sigma is not None else self.sigma
        return self.d1(sigma) - sigma * np.sqrt(self.T)
    
    def price(self, sigma=None):
        """
        Compute the Black-Scholes price for the option with continuous dividends.
        """
        sigma = sigma if sigma is not None else self.sigma
        d1 = self.d1(sigma)
        d2 = self.d2(sigma)
        if self.option_type == 'call':
            return self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type == 'put':
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    def delta(self):
        """
        Calculate and return the option's delta.
        """
        d1 = self.d1()
        if self.option_type == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(d1)
        elif self.option_type == 'put':
            return np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)
    
    def gamma(self):
        """
        Calculate and return the option's gamma.
        """
        d1 = self.d1()
        return np.exp(-self.q * self.T) * norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def theta(self):
        """
        Calculate and return the option's theta.
        """
        d1 = self.d1()
        d2 = self.d2()
        term1 = - (self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        if self.option_type == 'call':
            term2 = - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2) + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
            return term1 + term2
        elif self.option_type == 'put':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
            return term1 + term2
    
    def vega(self):
        """
        Calculate and return the option's vega.
        """
        d1 = self.d1()
        return self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T)
    
    def rho(self):
        """
        Calculate and return the option's rho.
        """
        d2 = self.d2()
        if self.option_type == 'call':
            return self.T * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type == 'put':
            return -self.T * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
    
    def implied_volatility_objective(self, sigma, market_price):
        """
        The objective function for the Newton-Raphson method to compute implied volatility.
        """
        return self.price(sigma) - market_price
    
    def implied_volatility(self, market_price):
        """
        Calculate the implied volatility given a market price using the Newton-Raphson method.
        """
        sigma_initial_guess = 0.2
        return newton(self.implied_volatility_objective, sigma_initial_guess, args=(market_price,))
    
    def summary(self):
        """
        Return a dictionary summary of the option's price and Greeks.
        """
        dic= {
            "Price": self.price(),
            "Delta": self.delta(),
            "Gamma": self.gamma(),
            "Theta": self.theta(),
            "Vega": self.vega(),
            "Rho": self.rho(),
            "Volatility": self.sigma
        }
        return pd.DataFrame(list(dic.items()), columns=["Metric", "Value"]).to_string(index=False)


