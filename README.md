# Stress-Testing-Workspace
📁 File Structure Summary

File	Purpose
main_pca_stress.py	             - Runs rolling stress test using PCA and outputs metrics
pca_module.py	                   - Performs PCA and reconstructs returns
scenario_generator.py	           - Applies stress to PCA components (single/multi)
impact_analysis.py               - Calculates portfolio VaR, ES, Drawdown, and sector contributions
returns_25_stocks.csv	           - Daily return matrix of 25 fixed stocks (2004–2024)
rolling_stress_with_deltas.csv   - Output: rolling baseline vs stressed metrics & sector deltas

🔁 Pipeline Summary

  1) PCA on rolling windows (e.g., 504 days)
  2) Compute baseline portfolio risk (no stress)
  3) Apply stress to PCA components
  4) Reconstruct stressed returns
  5) Compute stressed risk metrics
  6) Calculate delta (impact of stress) on total risk and sector-wise contributions
  7) Save all results to CSV for insights/visualization
