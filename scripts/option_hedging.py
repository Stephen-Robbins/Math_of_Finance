import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------- Black-Scholes Pricing Functions ----------

def black_scholes_call_price(S, E, r, T, sigma):
    """
    Compute the Black-Scholes price of a European call option.
    
    Parameters:
      S     : current stock price
      E     : strike price
      r     : risk-free rate
      T     : time to maturity (in years)
      sigma : volatility
      
    Returns:
      Call option price.
    """
    # At maturity, the option's value is its payoff.
    if T <= 0:
        return max(S - E, 0)
    
    d1 = (np.log(S/E) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - E * np.exp(-r*T) * norm.cdf(d2)
    return price

def call_delta(S, E, r, T, sigma):
    """
    Compute the Black-Scholes delta of a European call option.
    
    Parameters:
      S, E, r, T, sigma : same as above
      
    Returns:
      Call delta.
    """
    if T <= 0:
        return 1.0 if S > E else 0.0
    d1 = (np.log(S/E) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# ---------- Stock Path Simulation ----------

def simulate_stock_path(S0, r, sigma, T, n_steps):
    """
    Simulate a single stock price path using Geometric Brownian Motion.
    
    Parameters:
      S0     : initial stock price
      r      : risk-free rate (used as drift under risk-neutral measure)
      sigma  : volatility
      T      : time to maturity (years)
      n_steps: number of time steps (e.g., 252 for daily steps over one year)
      seed   : random seed (optional)
      
    Returns:
      Array of stock prices (length n_steps+1).
    """
    dt = T / n_steps
    S = np.zeros(n_steps + 1)
    S[0] = S0
    for t in range(n_steps):
        z = np.random.standard_normal()
        S[t+1] = S[t] * np.exp((r - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt)*z)
    return S

# ---------- Function 1: Replicating Portfolio Simulation ----------

def replicating_portfolio(S0, E, r, T, sigma, n_steps):
    """
    Simulate a self-financing replicating portfolio for a European call option.
    
    The option is priced (and hedged) using Black-Scholes with volatility sigma.
    At time 0 the portfolio is set to exactly match the option price.
    Then it is rebalanced at each time step to remain delta-neutral.
    
    Parameters:
      S0     : initial stock price
      E      : strike price
      r      : risk-free rate (annualized)
      T      : time to maturity (years)
      sigma  : volatility used for pricing/hedging (market implied)
      n_steps: number of time steps (e.g., daily steps over T)
      seed   : random seed (optional)
      
    Returns:
      time_grid         : array of time points (years)
      option_prices     : array of Black-Scholes option prices over time
      portfolio_values  : array of replicating portfolio values over time
      stock_path        : simulated stock price path
    """
    dt = T / n_steps
    stock_path = simulate_stock_path(S0, r, sigma, T, n_steps)
    
    option_prices    = np.zeros(n_steps + 1)
    portfolio_values = np.zeros(n_steps + 1)
    deltas           = np.zeros(n_steps + 1)
    bank_account     = np.zeros(n_steps + 1)
    
    # Initialization at t = 0.
    option_prices[0] = black_scholes_call_price(S0, E, r, T, sigma)
    deltas[0]        = call_delta(S0, E, r, T, sigma)
    portfolio_values[0] = option_prices[0]
    bank_account[0]     = portfolio_values[0] - deltas[0] * S0
    
    # Rebalance at each time step.
    for t in range(n_steps):
        bank_next   = bank_account[t] * np.exp(r*dt)
        stock_value = deltas[t] * stock_path[t+1]
        V_pre       = bank_next + stock_value
        
        time_remaining = T - (t+1)*dt
        option_prices[t+1] = black_scholes_call_price(stock_path[t+1], E, r, time_remaining, sigma)
        new_delta = call_delta(stock_path[t+1], E, r, time_remaining, sigma)
        deltas[t+1] = new_delta
        bank_account[t+1] = V_pre - new_delta * stock_path[t+1]
        portfolio_values[t+1] = V_pre  # self-financing: portfolio value remains unchanged by rebalancing.
    
    time_grid = np.linspace(0, T, n_steps + 1)
    return time_grid, option_prices, portfolio_values, stock_path

# ---------- Function 2: Arbitrage Simulation ----------
    
def arbitrage_simulation(S0, E, r, T, IV, sigma_R, n_steps):
    """Simulates arbitrage: buy market call, short replicating portfolio."""

    # --- Initial Option Prices ---
    initial_long_call_price = black_scholes_call_price(S0, E, r, T, IV)  # What you pay
    initial_short_call_price = black_scholes_call_price(S0, E, r, T, sigma_R) # Theoretical price


    # --- Initial Arbitrage Profit (and put it in the bank) ---
    arbitrage_profit = initial_short_call_price - initial_long_call_price
    

    _,option_prices, port_values, _=replicating_portfolio(S0, E, r, T, sigma_R, n_steps)

    diff= option_prices[-1]-port_values[-1]


    # Total P&L =  Initial Profit + Short Portfolio Value + Long Call Payoff
    realized_arbitrage_profit = arbitrage_profit +diff


    return arbitrage_profit, realized_arbitrage_profit

