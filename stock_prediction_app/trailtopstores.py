import pandas as pd

def get_top_performing_stores(start_date_str, end_date_str):
    # Load preprocessed data
    df = pd.read_csv('walmart_preprocessed_enhanced_v2.csv', parse_dates=['Date'])

    # Convert input dates
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Filter data within the date range
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df.loc[mask]

    # Check if the filtered dataframe is empty
    if df_filtered.empty:
        return {"error": "No sales data found in the given date range."}

    # Group by Store and calculate total sales
    store_sales = df_filtered.groupby('Store')['Weekly_Sales'].sum().reset_index()

    # Sort in descending order
    store_sales = store_sales.sort_values(by='Weekly_Sales', ascending=False)

    # Return as list of dictionaries
    results = []
    for index, row in store_sales.iterrows():
        results.append({
            "store": int(row['Store']),
            "total_sales": round(row['Weekly_Sales'], 2)
        })

    return results
