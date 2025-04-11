import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, r2_score
from pca_module import PCAReducer

# --- CONFIG ---
LOOKBACK = 252  # number of days in each training window
N_COMPONENTS = 10  # how many PCA features to use

# --- Load raw returns matrix ---
returns_df = pd.read_csv("stock_returns_matrix.csv")
returns_df.set_index("Date", inplace=True)
returns_df.index = pd.to_datetime(returns_df.index)
returns_df = returns_df.dropna()  # Drop any rows with missing values
returns_df = returns_df.asfreq('B', method='pad')  # Enforce business day frequency

# --- Step 1: Extract PCA latent features ---
pca = PCAReducer(returns_df.values, n_components=N_COMPONENTS)
pca_latent_df = pca.get_feature_dataframe(index=returns_df.index)

# --- Step 2: Compute target return (equal-weighted portfolio) and shift it ---
portfolio_returns = returns_df.mean(axis=1)
portfolio_returns.name = "portfolio_return"
portfolio_returns = portfolio_returns.shift(-1)
portfolio_returns = portfolio_returns.dropna()

# --- Align datasets ---
pca_latent_df = pca_latent_df.loc[portfolio_returns.index]
portfolio_returns = portfolio_returns.loc[pca_latent_df.index]

# --- Output containers ---
y_true = []
y_pred = []

# --- Rolling forecasting ---
start_time = time.time()
for i in range(LOOKBACK, len(pca_latent_df) - 1):
    X_window = pca_latent_df.iloc[i - LOOKBACK:i]
    y_window = portfolio_returns.iloc[i - LOOKBACK:i]

    # Step 3: Forecast next-day PCA components using AR(1)
    forecasted_components = []
    for pc in X_window.columns:
        series = X_window[pc].copy()
        series.index = pd.RangeIndex(start=0, stop=len(series), step=1)  # Avoid datetime index issues
        ar_model = AutoReg(series, lags=1).fit()
        forecast = ar_model.predict(start=len(series), end=len(series))
        forecasted_components.append(forecast.values[0])

    # Step 4: Train linear regressor on PCs → return
    model = LinearRegression().fit(X_window.values, y_window)

    # Step 5: Predict return using forecasted PCs
    return_hat = model.predict(np.array(forecasted_components).reshape(1, -1))

    # Store actual and predicted return for evaluation
    y_true.append(portfolio_returns.iloc[i + 1])
    y_pred.append(return_hat[0])

    if i%10==0:
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
end_time = time.time()

# --- Evaluate ---
y_true = np.array(y_true)
y_pred = np.array(y_pred)
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
directional_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

print("\n--- Rolling Forecast using PCA + AR(1) ---")
print(f"MSE: {mse:.6f}")
print(f"R^2 Score: {r2:.4f}")
print(f"Directional Accuracy: {directional_acc * 100:.2f}%")
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")

# --- Save predictions ---
pd.DataFrame({"Actual": y_true, "Predicted": y_pred},
             index=portfolio_returns.index[LOOKBACK+1:LOOKBACK+1+len(y_true)]).to_csv("rolling_pca_predictions.csv")
