import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler



# Load the dataset
df = pd.read_csv("Walmart.csv")

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Feature Engineering
# Extract year, month, week, and week of month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['Week_of_Month'] = df['Date'].dt.day // 7 + 1

# Create lag features for Weekly_Sales (previous 1 and 2 weeks)
df['Sales_Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
df['Sales_Lag_2'] = df.groupby('Store')['Weekly_Sales'].shift(2)

# Create rolling mean feature (7-week rolling mean)
df['Rolling_Mean'] = df.groupby('Store')['Weekly_Sales'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

# Handle missing values in lag and rolling mean (fill with group mean)
df['Sales_Lag_1'] = df.groupby('Store')['Sales_Lag_1'].transform(lambda x: x.fillna(x.mean()))
df['Sales_Lag_2'] = df.groupby('Store')['Sales_Lag_2'].transform(lambda x: x.fillna(x.mean()))
df['Rolling_Mean'] = df.groupby('Store')['Rolling_Mean'].transform(lambda x: x.fillna(x.mean()))

# Encode categorical variables
le_store = LabelEncoder()
df['Store_Encoded'] = le_store.fit_transform(df['Store'])

# Scale numerical features
scaler_temp = StandardScaler()
scaler_fuel = StandardScaler()
scaler_cpi = StandardScaler()
scaler_unemp = StandardScaler()
scaler_lag1 = StandardScaler()
scaler_lag2 = StandardScaler()
scaler_roll = StandardScaler()

df['Temperature_Scaled'] = scaler_temp.fit_transform(df[['Temperature']])
df['Fuel_Price_Scaled'] = scaler_fuel.fit_transform(df[['Fuel_Price']])
df['CPI_Scaled'] = scaler_cpi.fit_transform(df[['CPI']])
df['Unemployment_Scaled'] = scaler_unemp.fit_transform(df[['Unemployment']])
df['Sales_Lag_1_Scaled'] = scaler_lag1.fit_transform(df[['Sales_Lag_1']])
df['Sales_Lag_2_Scaled'] = scaler_lag2.fit_transform(df[['Sales_Lag_2']])
df['Rolling_Mean_Scaled'] = scaler_roll.fit_transform(df[['Rolling_Mean']])

# Create interaction features for holidays
df['Holiday_CPI_Scaled'] = df['Holiday_Flag'] * df['CPI_Scaled']
df['Holiday_Temperature_Scaled'] = df['Holiday_Flag'] * df['Temperature_Scaled']

# Encode specific holidays (basic example, assuming some holidays are known)
# For simplicity, assume holidays are marked by Holiday_Flag; extend with specific holiday encoding if needed
df['Specific_Holiday_Encoded'] = df['Holiday_Flag']  # Placeholder; can be enhanced with specific holiday names

# Save the preprocessed dataset
df.to_csv('walmart_preprocessed_enhanced_v2.csv', index=False)

print("Preprocessed dataset saved as 'walmart_preprocessed_enhanced_v2.csv'")