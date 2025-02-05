import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime

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