import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# Load model
model = joblib.load('best_model_gradient_boosting.pkl')

# Load raw data
df = pd.read_csv('Walmart.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Encode store
store_encoder = LabelEncoder()
df['Store_Encoded'] = store_encoder.fit_transform(df['Store'])

# Add temporal features
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Week'] = df['Date'].dt.isocalendar().week
df['Week_of_Month'] = df['Date'].apply(lambda d: (d.day - 1) // 7 + 1)

# Holiday name (simplified)
def get_holiday_name(row):
    if row['Holiday_Flag'] == 1:
        m, d = row['Date'].month, row['Date'].day
        if (m, d) in [(2, 12), (11, 26), (9, 5), (12, 25)]:
            return 'Super Bowl' if m == 2 else (
                'Thanksgiving' if m == 11 else (
                    'Labor Day' if m == 9 else 'Christmas'))
    return 'None'

df['Specific_Holiday'] = df.apply(get_holiday_name, axis=1)
df['Specific_Holiday_Encoded'] = LabelEncoder().fit_transform(df['Specific_Holiday'])

# Lag/rolling features
df = df.sort_values(['Store', 'Date'])
df['Sales_Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
df['Sales_Lag_2'] = df.groupby('Store')['Weekly_Sales'].shift(2)
df['Rolling_Mean'] = df.groupby('Store')['Weekly_Sales'].rolling(window=3).mean().reset_index(0, drop=True)

# Conditional holiday features
df['Holiday_CPI'] = df['CPI'].where(df['Holiday_Flag'] == 1, np.nan)
df['Holiday_Temperature'] = df['Temperature'].where(df['Holiday_Flag'] == 1, np.nan)

# Fill missing values
for col in ['Sales_Lag_1', 'Sales_Lag_2', 'Rolling_Mean', 'Holiday_CPI', 'Holiday_Temperature']:
    df[col] = df[col].fillna(df[col].median())

# Scaling
scale_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
              'Sales_Lag_1', 'Sales_Lag_2', 'Holiday_CPI',
              'Holiday_Temperature', 'Rolling_Mean']
scaler = MinMaxScaler()
df[[f"{c}_Scaled" for c in scale_cols]] = scaler.fit_transform(df[scale_cols])

# Final features
final_cols = [
    'Store_Encoded', 'Month', 'Holiday_Flag', 'Week_of_Month',
    'Specific_Holiday_Encoded', 'Temperature_Scaled', 'Fuel_Price_Scaled',
    'CPI_Scaled', 'Unemployment_Scaled', 'Sales_Lag_1_Scaled',
    'Sales_Lag_2_Scaled', 'Holiday_CPI_Scaled', 'Holiday_Temperature_Scaled',
    'Rolling_Mean_Scaled', 'Year', 'Week'
]

# Check final shape and sample
print(" Data shape for model:", df[final_cols].shape)
print(" Sample row for prediction:\n", df[final_cols].iloc[0])
