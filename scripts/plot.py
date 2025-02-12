# plot_functions.py

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.stats as stats
import pandas as pd
from scipy.stats import lognorm
from finance import black_scholes_price
import ipywidgets as widgets
from ipywidgets import interact

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



def plot_bs_option_sensitivity(option_type, S0, E, r, T, sigma):
    """
    Plot how the Black-Scholes option price changes as each parameter varies.
    Five subplots are generated (arranged in one row) showing the option price versus:
      - Underlying Price (S0)
      - Strike Price (K)
      - Risk-free Rate (r)
      - Time to Maturity (T)
      - Volatility (sigma)
      
    In each subplot, a vertical dashed line and a marker point indicate the original value.
    
    Parameters:
    - option_type: str, 'call' or 'put'
    - S0: float, current underlying price.
    - K: float, strike price.
    - r: float, risk-free interest rate.
    - T: float, time to maturity (in years).
    - sigma: float, volatility.
    
    Returns:
    - None (displays the plots)
    """
    # Define parameter ranges around the base values.
    S0_range    = np.linspace(0.5 * S0, 1.5 * S0, 100)
    K_range     = np.linspace(0.5 * E, 1.5 * E, 100)
    r_max       = r * 1.5 if r > 0 else 0.1
    r_range     = np.linspace(0, r_max, 100)
    T_range     = np.linspace(0.01, 2 * T, 100)   # start at 0.01 to avoid division by zero
    sigma_range = np.linspace(0.01, 2 * sigma, 100)

    # Create a 1x5 grid for a more condensed layout.
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # 1. Option Price vs Underlying Price (S0)
    price_vs_S0 = black_scholes_price(option_type, S0_range, E, r, T, sigma)
    axes[0].plot(S0_range, price_vs_S0, color='blue', linewidth=2)
    base_price = black_scholes_price(option_type, S0, E, r, T, sigma)
    axes[0].scatter([S0], [base_price], color='black', zorder=5)
    axes[0].axvline(S0, color='black', linestyle='--', alpha=0.7)
    axes[0].set_title('Underlying Price\n(S0)', fontsize=12)
    axes[0].set_xlabel('S_0')
    axes[0].set_ylabel('Option Price')
    axes[0].grid(True)
    
    # 2. Option Price vs Strike Price (K)
    price_vs_K = black_scholes_price(option_type, S0, K_range, r, T, sigma)
    axes[1].plot(K_range, price_vs_K, color='green', linewidth=2)
    base_price = black_scholes_price(option_type, S0, E, r, T, sigma)
    axes[1].scatter([E], [base_price], color='black', zorder=5)
    axes[1].axvline(E, color='black', linestyle='--', alpha=0.7)
    axes[1].set_title('Strike Price\n(K)', fontsize=12)
    axes[1].set_xlabel('E')
    axes[1].set_ylabel('Option Price')
    axes[1].grid(True)
    
    # 3. Option Price vs Risk-free Rate (r)
    price_vs_r = black_scholes_price(option_type, S0, E, r_range, T, sigma)
    axes[2].plot(r_range, price_vs_r, color='red', linewidth=2)
    base_price = black_scholes_price(option_type, S0, E, r, T, sigma)
    axes[2].scatter([r], [base_price], color='black', zorder=5)
    axes[2].axvline(r, color='black', linestyle='--', alpha=0.7)
    axes[2].set_title('Risk-free Rate\n(r)', fontsize=12)
    axes[2].set_xlabel('r')
    axes[2].set_ylabel('Option Price')
    axes[2].grid(True)
    
    # 4. Option Price vs Time to Maturity (T)
    price_vs_T = black_scholes_price(option_type, S0, E, r, T_range, sigma)
    axes[3].plot(T_range, price_vs_T, color='purple', linewidth=2)
    base_price = black_scholes_price(option_type, S0, E, r, T, sigma)
    axes[3].scatter([T], [base_price], color='black', zorder=5)
    axes[3].axvline(T, color='black', linestyle='--', alpha=0.7)
    axes[3].set_title('Time to Maturity\n(T)', fontsize=12)
    axes[3].set_xlabel('T (years)')
    axes[3].set_ylabel('Option Price')
    axes[3].grid(True)
    
    # 5. Option Price vs Volatility (sigma)
    price_vs_sigma = black_scholes_price(option_type, S0, E, r, T, sigma_range)
    axes[4].plot(sigma_range, price_vs_sigma, color='orange', linewidth=2)
    base_price = black_scholes_price(option_type, S0, E, r, T, sigma)
    axes[4].scatter([sigma], [base_price], color='black', zorder=5)
    axes[4].axvline(sigma, color='black', linestyle='--', alpha=0.7)
    axes[4].set_title('Volatility\n(sigma)', fontsize=12)
    axes[4].set_xlabel('sigma')
    axes[4].set_ylabel('Option Price')
    axes[4].grid(True)
    
    plt.tight_layout()
    plt.show()
def interactive_bs_s0_plot(option_type="call", base_S0=100):
    """
    Creates an interactive plot of the option price (using the Black-Scholes formula)
    as the underlying price S0 varies, with adjustable parameters for the other variables.
    
    The plot shows:
      - The option price curve versus S0 (ranging from 0.5*base_S0 to 1.5*base_S0)
      - A marker and a dashed vertical line indicating the base S0 value.
    
    You can interactively adjust:
      - Strike Price (K)
      - Risk-free Rate (r)
      - Time to Maturity (T)
      - Volatility (sigma)
    
    Parameters:
    - option_type: str, either 'call' or 'put' (default is 'call')
    - base_S0: float, the base underlying price at which a marker is shown (default is 100)
    
    To use in Google Colab, simply run:
    
        from bs_interactive import interactive_bs_s0_plot
        interactive_bs_s0_plot()
    """
    def plot_function(K, r, T, sigma):
        # Define a range for S0 around the base value.
        S0_range = np.linspace(0.5 * base_S0, 1.5 * base_S0, 200)
        prices = black_scholes_price(option_type, S0_range, K, r, T, sigma)
        base_price = black_scholes_price(option_type, base_S0, K, r, T, sigma)
        
        plt.figure(figsize=(8, 5))
        plt.plot(S0_range, prices, color='blue', lw=2, label=f"{option_type.title()} Price")
        plt.scatter([base_S0], [base_price], color='black', zorder=5, label=f"Base S₀ = {base_S0}")
        plt.axvline(base_S0, color='black', linestyle='--', alpha=0.7)
        plt.title(f"Option Price vs Underlying Price (S₀)\n(K = {K}, r = {r}, T = {T}, σ = {sigma})", fontsize=14)
        plt.xlabel("Underlying Price (S₀)", fontsize=12)
        plt.ylabel("Option Price", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Create interactive widgets for K, r, T, and sigma.
    interact(
        plot_function,
        K=widgets.FloatSlider(value=100, min=50, max=150, step=1, description='Strike (K)'),
        r=widgets.FloatSlider(value=0.05, min=0.0, max=0.2, step=0.005, description='Risk-free (r)'),
        T=widgets.FloatSlider(value=1, min=0.0, max=5, step=0.01, description='Time (T)'),
        sigma=widgets.FloatSlider(value=0.2, min=0.1, max=1.0, step=0.01, description='Volatility (σ)')
    )
