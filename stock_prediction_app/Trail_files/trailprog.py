import pandas as pd
import numpy as np
import joblib



def forecast_weekly_sales(start_date_str, end_date_str, store_number):
    # --- Step 0: Load model and preprocessed sales data ---
    model_path = 'best_model_gradient_boosting.pkl'
    loaded_model = joblib.load(model_path)
    df = pd.read_csv('walmart_preprocessed_enhanced_v2.csv', parse_dates=['Date'])

    features = [
        'Store_Encoded', 'Year', 'Month', 'Week', 'Week_of_Month',
        'Holiday_Flag', 'Temperature_Scaled', 'Fuel_Price_Scaled', 'CPI_Scaled',
        'Unemployment_Scaled', 'Sales_Lag_1_Scaled', 'Sales_Lag_2_Scaled',
        'Holiday_CPI_Scaled', 'Holiday_Temperature_Scaled',
        'Rolling_Mean_Scaled', 'Specific_Holiday_Encoded'
    ]

    # --- Step 1: Convert input dates ---
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # --- Step 2: Generate weekly blocks ---
    forecast_dates = []
    current = start_date
    while current + pd.Timedelta(days=6) <= end_date:
        forecast_dates.append(current)
        current += pd.Timedelta(days=7)
    if current <= end_date:
        forecast_dates.append(current)

    # --- Step 3: Calculate weekday contribution ratios ---
    df['Weekday'] = df['Date'].dt.dayofweek
    weekday_avg = df.groupby('Weekday')['Weekly_Sales'].mean()
    weekday_share = weekday_avg / weekday_avg.sum()
    weekday_share = weekday_share.reindex(range(7), fill_value=0)

    # Get Store_Encoded from original dataframe
    store_map = df[['Store', 'Store_Encoded']].drop_duplicates()

    # Convert user input to int (subtract 1 if needed — as you mentioned)
    store_id = int(store_number)

    # Validate store exists
    if store_id not in store_map['Store'].values:
        raise ValueError(f"Store {store_id + 1} not found in dataset.")

    # Fetch Store_Encoded value
    store_encoded = store_map.loc[store_map['Store'] == store_id, 'Store_Encoded'].values[0]


    # --- Step 4: Handle partial final week ---
    last_week_start = forecast_dates[-1]
    last_week_end = last_week_start + pd.Timedelta(days=6)
    if last_week_end > end_date:
        covered_dates = pd.date_range(start=last_week_start, end=end_date)
        covered_weekdays = covered_dates.dayofweek
        partial_week_factor = weekday_share.loc[covered_weekdays].sum()
    else:
        partial_week_factor = 1.0

    # --- Step 5: Create future DataFrame ---
    future_df = pd.DataFrame(index=range(len(df), len(df) + len(forecast_dates)))
    future_df['Date'] = forecast_dates
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Week'] = future_df['Date'].dt.isocalendar().week
    future_df['Week_of_Month'] = future_df['Date'].dt.day // 7 + 1

    # Fill with latest known values
    # Filter only the data for the selected store
    df_store = df[df['Store'] == store_id].copy()

    if df_store.empty:
        raise ValueError(f"No historical data available for store {store_id}")

    # Get last known row for this specific store
    last_row = df_store.sort_values('Date').iloc[-1]
    if last_row.isnull().any():
        raise ValueError(f"Last known values for store {store_id} contain missing values.")


    future_df['Holiday_Flag'] = 0
    future_df['Temperature_Scaled'] = last_row['Temperature_Scaled']
    future_df['Fuel_Price_Scaled'] = last_row['Fuel_Price_Scaled']
    future_df['CPI_Scaled'] = last_row['CPI_Scaled']
    future_df['Unemployment_Scaled'] = last_row['Unemployment_Scaled']
    future_df['Sales_Lag_1_Scaled'] = last_row['Sales_Lag_1_Scaled']
    future_df['Sales_Lag_2_Scaled'] = last_row['Sales_Lag_2_Scaled']
    future_df['Rolling_Mean_Scaled'] = last_row['Rolling_Mean_Scaled']
    future_df['Store_Encoded'] = store_encoded
    future_df['Holiday_CPI_Scaled'] = last_row['CPI_Scaled']
    future_df['Holiday_Temperature_Scaled'] = last_row['Temperature_Scaled']
    future_df['Specific_Holiday_Encoded'] = 0


    # --- Step 6: Predict ---
    X_future = future_df[features]
    predictions = loaded_model.predict(X_future)

    # --- Step 7: Adjust final week for partial coverage ---
    weekday_fixed_ratios = {0: 0.13, 1: 0.14, 2: 0.13, 3: 0.14, 4: 0.15, 5: 0.17, 6: 0.14}
    is_partial_week = False
    original_pred = predictions[-1]

    if last_week_end > end_date:
        covered_dates = pd.date_range(start=last_week_start, end=end_date)
        covered_weekdays = covered_dates.dayofweek
        partial_week_factor = sum(weekday_fixed_ratios.get(day, 0) for day in covered_weekdays)
        predictions[-1] = original_pred * partial_week_factor
        is_partial_week = True

    # --- Step 8: Return results with weekend (end of week) ---
    result = []
    for i, (date, pred) in enumerate(zip(future_df['Date'], predictions)):
        week_start = date
        week_end = min(date + pd.Timedelta(days=6), end_date)

        entry = {
            'From': week_start.strftime('%Y-%m-%d'),
            'Till': week_end.strftime('%Y-%m-%d'),
            'sales_prediction': round(float(pred), 2)
        }

        if i == len(predictions) - 1 and is_partial_week:
            entry['note'] = f"Partial week scaled from raw ${original_pred:.2f} using factor {partial_week_factor:.2f}"

        result.append(entry)

    return result

def debug_forecast_output(forecast_result):
    print("\n========== Weekly Forecast Debug ==========\n")
    total_sales = 0.0

    for i, week in enumerate(forecast_result, 1):
        print(f"Week {i}:")
        print(f"    Start Date : {week['From']}")
        print(f"    End Date   : {week['Till']}")
        print(f"   Predicted Sales: ${week['sales_prediction']:.2f}")

        if 'note' in week:
            print(f"    Note: {week['note']}")

        total_sales += week['sales_prediction']
        print("--------------------------------------")

    print(f"\n Total Forecasted Sales: ${total_sales:.2f}")
    print("==========================================\n")


# Step 1: Get the forecast
start_date = '2025-08-14'
end_date = '2025-09-08' 
store_number = 2  # Example store number, adjust as needed
forecast_result = forecast_weekly_sales(start_date, end_date, store_number)

# Step 2: Pass it to the debug function
debug_forecast_output(forecast_result)