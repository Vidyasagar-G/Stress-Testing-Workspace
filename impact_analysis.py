import numpy as np

def compute_portfolio_returns(returns_matrix, weights):
    return np.dot(returns_matrix, weights)

def compute_var(portfolio_returns, confidence_level=0.95):
    sorted_returns = np.sort(portfolio_returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    return abs(sorted_returns[index])

def compute_expected_shortfall(portfolio_returns, confidence_level=0.95):
    sorted_returns = np.sort(portfolio_returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    tail_losses = sorted_returns[:index]
    return abs(np.mean(tail_losses))

def compute_drawdown(portfolio_returns):
    cumulative = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return np.min(drawdown)

def compute_sector_contributions(returns_matrix, weights, sector_labels):
    unique_sectors = sorted(set(sector_labels))
    sector_contributions = {}
    for sector in unique_sectors:
        idx = [i for i, label in enumerate(sector_labels) if label == sector]
        sector_returns = returns_matrix[:, idx]
        sector_weights = weights[idx]
        sector_contributions[sector] = np.dot(sector_returns, sector_weights)
    return sector_contributions
