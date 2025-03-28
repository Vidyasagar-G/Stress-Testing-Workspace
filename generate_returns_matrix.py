import os
import pandas as pd
import numpy as np

# Path to your main data folder
data_dir = "C:\All data\IITG\Acads\Eighth semester\MA499 Project II\stress_testing_code\Indian_Stock_Data"

# Output dataframe
returns_matrix = pd.DataFrame()

# Loop through each sector folder
for sector in os.listdir(data_dir):
    sector_path = os.path.join(data_dir, sector)
    
    if not os.path.isdir(sector_path):
        continue  # Skip files, only process folders
    
    # Loop through each stock CSV in the sector
    for file in os.listdir(sector_path):
        if not file.endswith('.csv'):
            continue
        
        file_path = os.path.join(sector_path, file)
        stock_name = f"{sector}_{os.path.splitext(file)[0]}"
        
        # Read stock data
        df = pd.read_csv(file_path, parse_dates=['Date'])
        
        # Compute daily returns
        df['Return'] = df['Close'].pct_change()  # or use np.log(df['Close']).diff() for log returns
        df = df.dropna(subset=['Return'])  # drop the first row with NaN
        
        # Keep only Date and Return
        df = df[['Date', 'Return']]
        df = df.set_index('Date')
        df = df.rename(columns={'Return': stock_name})
        
        # Join into master matrix
        if returns_matrix.empty:
            returns_matrix = df
        else:
            returns_matrix = returns_matrix.join(df, how='outer')

# Transpose to get 25 x T shape
returns_matrix = returns_matrix.T.sort_index()

# Fill missing values if needed
returns_matrix = returns_matrix.fillna(0)  # or use returns_matrix.dropna(axis=1)

returns_matrix = returns_matrix.T

# Save to CSV
returns_matrix.to_csv('stock_returns_matrix.csv')

# Final shape: (25 stocks) x (T days)
print("Shape of final return matrix:", returns_matrix.shape)
