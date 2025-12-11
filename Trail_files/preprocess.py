import sys
from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os
import joblib
from flask import jsonify
import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_walmart_data(input_csv='Walmart.csv', output_csv='walmart_preprocessed_enhanced_v2.csv'):
    try:
        df = pd.read_csv(input_csv)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

        # Sort before applying rolling and lag operations
        df.sort_values(['Store', 'Date'], inplace=True)

        df['Year'] = df['Date'].dt.year
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Week_of_Month'] = df['Date'].dt.day // 7 + 1
        df['Sales_Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
        df['Sales_Lag_2'] = df.groupby('Store')['Weekly_Sales'].shift(2)
        df['Holiday_CPI'] = df['Holiday_Flag'] * df['CPI']
        df['Holiday_Temperature'] = df['Holiday_Flag'] * df['Temperature']
        df['Rolling_Mean'] = df.groupby('Store')['Weekly_Sales'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )

        if 'Specific_Holiday' not in df.columns:
            df['Specific_Holiday'] = 'None'

        le_store = LabelEncoder()
        df['Store_Encoded'] = le_store.fit_transform(df['Store'])

        le_holiday = LabelEncoder()
        df['Specific_Holiday_Encoded'] = le_holiday.fit_transform(df['Specific_Holiday'])

        df['Weekly_Sales_Capped'] = df.groupby('Store')['Weekly_Sales'].transform(
            lambda x: x.clip(upper=x.mean() + 3 * x.std())
        )

        scaler = StandardScaler()
        numeric_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                        'Sales_Lag_1', 'Sales_Lag_2', 'Holiday_CPI',
                        'Holiday_Temperature', 'Rolling_Mean']

        df[[f'{col}_Scaled' for col in numeric_cols]] = scaler.fit_transform(df[numeric_cols])

        # Drop any rows with NaN AFTER all feature engineering
        df.dropna(inplace=True)

        df.to_csv(output_csv, index=False)
        print(f" Preprocessing complete. Saved to '{output_csv}'", flush=True)

        # Optional check
        last_rows = df.sort_values(['Store', 'Date']).groupby('Store').tail(1)
        if last_rows.isnull().any().any():
            print(" Warning: Some last rows per store still have missing values.")

        return df

    except Exception as e:
        print(f" Error during preprocessing: {e}", flush=True)
        import sys
        sys.exit(1)


# Ensure the function is called when script runs
if __name__ == "__main__":
    preprocess_walmart_data()
