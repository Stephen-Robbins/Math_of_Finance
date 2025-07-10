import numpy as np

def calculate_optimal_bets(odds, total_money):

    # Convert odds to numpy array for easier calculations
    odds = np.array(odds)

    # Calculate the denominator for optimal bets formula
    denominator = np.sum(1 / (1 + odds))

    # Calculate optimal bets for each outcome
    optimal_bets = total_money / denominator * (1 / (1 + odds))

    # Calculate the guaranteed profit
    profit = total_money * (np.prod(1 + odds) / np.sum(np.prod(1 + odds) / (1 + odds)) - 1)

    return optimal_bets, profit

def probability_to_odds(probability):
    """
    Convert probability to odds.

    Args:
    probability (float): A value between 0 and 1 representing the probability.

    Returns:
    float: The corresponding odds.
    """
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1")
    return (1 - probability) / probability

def odds_to_probability(odds):
    """
    Convert odds to probability.

    Args:
    odds (float): The odds value.

    Returns:
    float: The corresponding probability between 0 and 1.
    """
    return 1 / (odds + 1)

