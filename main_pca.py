import numpy as np
import pandas as pd
from pca_module import PCAReducer
from scenario_generator import PCAScenarioGenerator
from impact_analysis import (
    compute_portfolio_returns,
    compute_var,
    compute_expected_shortfall,
    compute_drawdown,
    compute_sector_contributions
)

# === USER INPUTS === #
returns_df = pd.read_csv("stock_returns_matrix.csv", index_col=0)  # shape (T, N)
returns_matrix = returns_df.values
dates = returns_df.index.tolist()

# Assuming your return matrix has stock names as columns in the format 'Sector_Company'
stock_names = returns_df.columns  # or rows if transposed
sector_labels = [name.split('_')[0] for name in stock_names]


weights = np.ones(25) / 25

# === ROLLING CONFIG === #
window_size = 252  # ~2 years
step_size = 21     # ~1 month

# === STRESS CONFIG === #
# Option A: Single component (e.g., PC1)
stress_mode = "multi"  # "single" or "multi"
pc_index = 0            # Only used if stress_mode == "single"
sigma_multiplier = 2.0

# Option B: Multi-component stress (±σ on multiple PCs)
shift_vector = [2.0, -1.5, 1.0, 0.5, -0.5]  # For the first 5 PCs

results = []

for start_idx in range(0, len(returns_matrix) - window_size + 1, step_size):
    end_idx = start_idx + window_size
    window_returns = returns_matrix[start_idx:end_idx]
    window_dates = (dates[start_idx], dates[end_idx - 1])

    # === STEP 1: PCA ===
    pca = PCAReducer(window_returns, n_components=5)
    components = pca.get_components()

    # print("components: ",components.shape)

    # === BASELINE (Unstressed) ===
    base_returns = pca.inverse_transform(components)
    base_portfolio_returns = compute_portfolio_returns(base_returns, weights)
    base_var = compute_var(base_portfolio_returns)
    base_es = compute_expected_shortfall(base_portfolio_returns)
    base_dd = compute_drawdown(base_portfolio_returns)
    base_sector = compute_sector_contributions(base_returns, weights, sector_labels)

    # === STEP 2: STRESS SCENARIO ===
    scenario_gen = PCAScenarioGenerator(components)

    if stress_mode == "single":
        stressed_components = scenario_gen.apply_single_component_shift(
            pc_index=pc_index,
            sigma_multiplier=sigma_multiplier
        )
    elif stress_mode == "multi":
        stressed_components = scenario_gen.apply_multi_component_shift(shift_vector)

    # === STEP 3: RECONSTRUCT STRESSED RETURNS ===
    stressed_returns = pca.inverse_transform(stressed_components)

    # === STEP 4: STRESSED METRICS ===
    stress_portfolio_returns = compute_portfolio_returns(stressed_returns, weights)
    stress_var = compute_var(stress_portfolio_returns)
    stress_es = compute_expected_shortfall(stress_portfolio_returns)
    stress_dd = compute_drawdown(stress_portfolio_returns)
    stress_sector = compute_sector_contributions(stressed_returns, weights, sector_labels)

    # === STEP 5: DELTA CALCULATION ===
    delta_var = stress_var - base_var
    delta_es = stress_es - base_es
    delta_dd = stress_dd - base_dd
    delta_sector = {
        sec: np.mean(stress_sector[sec]) - np.mean(base_sector[sec])
        for sec in base_sector.keys()
    }

    explained_variance = pca.get_explained_variance().tolist()
    explained_variance_dict = {f"explained_variance_{i+1}": val for i, val in enumerate(explained_variance)}

    # === RECORD EVERYTHING ===
    results.append({
        "window_start": window_dates[0],
        "window_end": window_dates[1],
        "base_VaR_95": base_var,
        "stress_VaR_95": stress_var,
        "delta_VaR_95": delta_var,
        "base_ES_95": base_es,
        "stress_ES_95": stress_es,
        "delta_ES_95": delta_es,
        "base_Drawdown": base_dd,
        "stress_Drawdown": stress_dd,
        "delta_Drawdown": delta_dd,
        "base_sector": {k: float(np.mean(v)) for k, v in base_sector.items()},
        "stress_sector": {k: float(np.mean(v)) for k, v in stress_sector.items()},
        "delta_sector": delta_sector,
        "explained_variance": explained_variance_dict
    })

# === EXPORT RESULTS === #
df = pd.json_normalize(results)
if stress_mode == "single":
    df.to_csv("rolling_stress_with_deltas.csv", index=False)
elif stress_mode == "multi":
    df.to_csv("rolling_stress_with_deltas_multiPC.csv", index=False)


print("\nStress testing with baseline comparison complete.")
print("Results saved to csv file.")