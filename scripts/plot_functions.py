# plot_functions.py

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.stats as stats
import pandas as pd
from scipy.stats import lognorm


def plot_gbm_path(S0=100, mu=0.05, sigma=0.2, T=1, N=252, seed=None, n_paths=1):
    """
    Simulate and plot one or multiple geometric Brownian motion (GBM) paths.
    
    Parameters:
    - S0: Initial stock price (default=100)
    - mu: Drift (expected return) (default=0.05)
    - sigma: Volatility (default=0.2)
    - T: Total time in years (default=1)
    - N: Number of time steps (default=252, e.g., daily steps over a year)
    - seed: Random seed for reproducibility (default=None)
    - n_paths: Number of GBM paths to simulate (default=1)
    
    Returns:
    - t: Array of time points.
    - S_paths: A 2D array of simulated paths with shape (n_paths, N+1).
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    t = np.linspace(0, T, N+1)
    
    # Prepare an array to store all paths
    S_paths = np.zeros((n_paths, N+1))
    
    # Simulate each path
    for path in range(n_paths):
        S = np.zeros(N+1)
        S[0] = S0
        # Generate standard normal random variables for this path
        Z = np.random.normal(size=N)
        for i in range(1, N+1):
            S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i-1])
        S_paths[path] = S

    # Plotting
    plt.figure(figsize=(10, 6))
    for path in range(n_paths):
        plt.plot(t, S_paths[path], lw=1.5, label=f'Path {path+1}' if n_paths <= 5 else None)
    
    plt.xlabel('Time (years)')
    plt.ylabel('Price')
    plt.title(f'Simulated Geometric Brownian Motion Path{"s" if n_paths > 1 else ""}')
    if n_paths <= 5:
        plt.legend()
    plt.grid(True)
    plt.show()
    
    return t, S_paths


def plot_stock_path(ticker, start, end):
    """
    Download and plot real stock data from Yahoo Finance.
    
    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL')
    - start: Start date in 'YYYY-MM-DD' format (e.g., '2020-01-01')
    - end: End date in 'YYYY-MM-DD' format (e.g., '2021-01-01')
    """
    # Download stock data using yfinance
    data = yf.download(ticker, start=start, end=end)
    
    # Check if data was successfully retrieved
    if data.empty:
        print("No data found. Please check the ticker symbol and date range.")
        return None
    
    # Plotting the closing prices
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label=f'{ticker} Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} Stock Price from {start} to {end}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Optionally return the DataFrame for further analysis
    return data

def plot_log_returns_bell_curve(ticker, start, end, bins=50):
    """
    Downloads historical stock data, computes log returns, and plots:
      - A histogram of the log returns.
      - A bell curve (normal distribution) fitted to the log returns.
      - Annotations displaying the mean and variance of the log returns.
    
    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL').
    - start: Start date in 'YYYY-MM-DD' format.
    - end: End date in 'YYYY-MM-DD' format.
    - bins: Number of bins for the histogram (default is 50).
    """
    # Download historical data using yfinance
    data = yf.download(ticker, start=start, end=end)
    
    if data.empty:
        print("No data found. Please check the ticker symbol and date range.")
        return None

    # Calculate log returns based on closing prices.
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()  # Remove the NaN value from the first row.

    # Calculate statistics: mean, variance, and standard deviation.
    mean_val = data['Log_Returns'].mean()
    var_val = data['Log_Returns'].var()
    std_val = data['Log_Returns'].std()

    # Plotting the histogram of log returns.
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(data['Log_Returns'], bins=bins, density=True, alpha=0.6,
                                     color='skyblue', edgecolor='black',
                                     label="Log Returns Histogram")

    # Generate x values over the range of the histogram.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    # Compute the normal probability density function (PDF) based on the log returns' mean and standard deviation.
    p = stats.norm.pdf(x, mean_val, std_val)
    
    # Plot the bell curve.
    plt.plot(x, p, 'r', linewidth=2, label="Fitted Normal Distribution")
    
    # Add title and labels.
    plt.title(f'Log Returns and Fitted Normal Distribution for {ticker}')
    plt.xlabel('Log Returns')
    plt.ylabel('Density')
    
    # Annotate the plot with the calculated mean and variance.
    plt.text(0.05, 0.95, f'Mean: {mean_val:.5f}\nVariance: {var_val:.5f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add a legend and grid.
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Optionally return the data for further analysis.
    return data



def plot_log_normal_distribution(mu, S0, sigma, T, num_points=1000):
    """
    Plot the log-normal probability density function (PDF) for the stock price S_T 
    under geometric Brownian motion.
    
    The stock price S_T is modeled as:
        S_T = S0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    where Z ~ N(0,1). Thus, S_T is log-normally distributed.
    
    Parameters:
    - mu: Drift (mean return) parameter.
    - S0: Initial stock price.
    - sigma: Volatility (diffusion) parameter.
    - T: Time horizon (in years, for example).
    - num_points: Number of points to use for plotting the PDF (default: 1000).
    
    The function will plot the PDF of S_T and display the plot.
    """
    # Calculate the shape parameter (sigma * sqrt(T)) and scale parameter.
    shape = sigma * np.sqrt(T)
    scale = S0 * np.exp((mu - 0.5 * sigma**2) * T)
    
    # Create a range of S_T values. We choose a range that covers the bulk of the probability.
    # For example, from near 0 to an upper limit based on the scale.
    # Adjust the upper limit multiplier as needed for your data.
    S_min = 0.001  # avoid zero to prevent division by zero in lognorm.pdf
    S_max = scale * 4  # this factor can be tuned as desired
    S_values = np.linspace(S_min, S_max, num_points)
    
    # Compute the PDF using scipy.stats.lognorm
    pdf_values = lognorm.pdf(S_values, s=shape, scale=scale)
    
    # Plot the PDF
    plt.figure(figsize=(10, 6))
    plt.plot(S_values, pdf_values, 'r-', lw=2, label='Log-Normal PDF')
    plt.xlabel(r'$S_T$')
    plt.ylabel('Probability Density')
    plt.title(r'Log-Normal Distribution of $S_T$ under Geometric Brownian Motion')
    plt.legend()
    plt.grid(True)
    
    # Display parameters on the plot
    plt.text(0.65, 0.85,
             f"$S_0$ = {S0}\n$\mu$ = {mu}\n$\sigma$ = {sigma}\n$T$ = {T}",
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='wheat', alpha=0.5))
    
    plt.show()
