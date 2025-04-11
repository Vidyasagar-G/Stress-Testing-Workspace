import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, r2_score
from autoencoder_module import AutoencoderReducer

# --- CONFIG ---
LOOKBACK = 252  # number of days in each training window
LATENT_DIM = 10  # number of AE latent features to use

# --- Load raw returns matrix ---
returns_df = pd.read_csv("stock_returns_matrix.csv")
returns_df.set_index("Date", inplace=True)
returns_df.index = pd.to_datetime(returns_df.index)
returns_df = returns_df.dropna()
returns_df = returns_df.asfreq('B', method='pad')

# --- Train AE on full data to initialize encoder ---
ae_model = AutoencoderReducer(returns_matrix=returns_df.values, encoding_dim=LATENT_DIM)
latent_df_full = ae_model.get_feature_dataframe(index=returns_df.index)

# --- Step 2: Compute target return (equal-weighted portfolio) and shift it ---
portfolio_returns = returns_df.mean(axis=1)
portfolio_returns.name = "portfolio_return"
portfolio_returns = portfolio_returns.shift(-1)
portfolio_returns = portfolio_returns.dropna()

# --- Align datasets ---
latent_df_full = latent_df_full.loc[portfolio_returns.index]
portfolio_returns = portfolio_returns.loc[latent_df_full.index]

# --- Output containers ---
y_true = []
y_pred = []

# --- Rolling forecasting ---
start_time = time.time()
for i in range(LOOKBACK, len(latent_df_full) - 1):
    X_window = latent_df_full.iloc[i - LOOKBACK:i]
    y_window = portfolio_returns.iloc[i - LOOKBACK:i]

    # Step 3: Forecast next-day AE latent features using AR(1)
    forecasted_latents = []
    for z in X_window.columns:
        series = X_window[z].copy()
        series.index = pd.RangeIndex(start=0, stop=len(series), step=1)
        ar_model = AutoReg(series, lags=1).fit()
        forecast = ar_model.predict(start=len(series), end=len(series))
        forecasted_latents.append(forecast.values[0])

    # Step 4: Train linear regressor on AE latents → return
    model = LinearRegression().fit(X_window.values, y_window)

    # Step 5: Predict return using forecasted latent vector
    return_hat = model.predict(np.array(forecasted_latents).reshape(1, -1))

    y_true.append(portfolio_returns.iloc[i + 1])
    y_pred.append(return_hat[0])

    if i % 10 == 0:
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
end_time = time.time()

# --- Evaluate ---
y_true = np.array(y_true)
y_pred = np.array(y_pred)
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
directional_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

print("\n--- Rolling Forecast using AE + AR(1) ---")
print(f"MSE: {mse:.6f}")
print(f"R^2 Score: {r2:.4f}")
print(f"Directional Accuracy: {directional_acc * 100:.2f}%")
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")

# --- Save predictions ---
pd.DataFrame({"Actual": y_true, "Predicted": y_pred},
             index=portfolio_returns.index[LOOKBACK+1:LOOKBACK+1+len(y_true)]).to_csv("rolling_ae_predictions.csv")
