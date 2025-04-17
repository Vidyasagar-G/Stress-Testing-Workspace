import time
import numpy as np
import pandas as pd
from vae_module import VariationalAutoencoderReducer
from scenario_generator import AEScenarioGenerator
from impact_analysis import (
    compute_portfolio_returns,
    compute_var,
    compute_expected_shortfall,
    compute_drawdown,
    compute_sector_contributions
)

# === USER INPUTS === #
returns_df = pd.read_csv("stock_returns_matrix.csv", index_col=0)
returns_matrix = returns_df.values
dates = returns_df.index.tolist()

stock_names = returns_df.columns
sector_labels = [name.split('_')[0] for name in stock_names]

weights = np.ones(25) / 25  # Equal weights

# === ROLLING CONFIG === #
window_size = 504  # 2-year window
step_size = 21     # 1-month step

# === STRESS CONFIG === #
latent_index = 0
sigma_multiplier = 2.0
shift_vector = [2.0, -1.5, 1.0, 0.5, -0.5]

results_single = []
results_multi = []

for start_idx in range(0, len(returns_matrix) - window_size + 1, step_size):
    start_time = time.time()
    end_idx = start_idx + window_size
    window_returns = returns_matrix[start_idx:end_idx]
    window_dates = (dates[start_idx], dates[end_idx - 1])

    # === STEP 1: Train VAE === #
    vae = VariationalAutoencoderReducer(window_returns, latent_dim=5, epochs=100, batch_size=32)
    latent_mean = vae.get_z_mean()  # deterministic latent features

    # === BASELINE === #
    base_returns = vae.inverse_transform(latent_mean)
    base_portfolio = compute_portfolio_returns(base_returns, weights)
    base_var = compute_var(base_portfolio)
    base_es = compute_expected_shortfall(base_portfolio)
    base_dd = compute_drawdown(base_portfolio)
    base_sector = compute_sector_contributions(base_returns, weights, sector_labels)

    # Save loss plot periodically
    if start_idx % 20 == 0:
        vae.plot_reconstruction_loss(save_path=f"vae_loss_{window_dates[0]}_{window_dates[1]}.png")

    # === STEP 2: Stress Scenarios === #
    scenario_gen = AEScenarioGenerator(latent_mean)

    for mode in ["single", "multi"]:
        if mode == "single":
            stressed_latent = scenario_gen.apply_single_latent_shift(
                ae_index=latent_index,
                sigma_multiplier=sigma_multiplier
            )
        elif mode == "multi":
            stressed_latent = scenario_gen.apply_multi_latent_shift(shift_vector)

        # === STEP 3: RECONSTRUCTION === #
        stressed_returns = vae.inverse_transform(stressed_latent)

        # === STEP 4: METRICS === #
        stress_portfolio = compute_portfolio_returns(stressed_returns, weights)
        stress_var = compute_var(stress_portfolio)
        stress_es = compute_expected_shortfall(stress_portfolio)
        stress_dd = compute_drawdown(stress_portfolio)
        stress_sector = compute_sector_contributions(stressed_returns, weights, sector_labels)

        delta_var = stress_var - base_var
        delta_es = stress_es - base_es
        delta_dd = stress_dd - base_dd
        delta_sector = {
            sec: np.mean(stress_sector[sec]) - np.mean(base_sector[sec])
            for sec in base_sector
        }

        record = {
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
            "delta_sector": delta_sector
        }

        if mode == "single":
            results_single.append(record)
        elif mode == "multi":
            results_multi.append(record)

    print(f"[✓] Completed window {start_idx+1}: Time = {round(time.time() - start_time, 2)} sec")

# === EXPORT RESULTS === #
df_single = pd.json_normalize(results_single)
df_multi = pd.json_normalize(results_multi)

df_single.to_csv("rolling_vae_stress_with_deltas_single.csv", index=False)
df_multi.to_csv("rolling_vae_stress_with_deltas_multi.csv", index=False)

print("\nVAE-based stress testing complete.")
print("Results saved to 'rolling_vae_stress_with_deltas_*.csv'.")
