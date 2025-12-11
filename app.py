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
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Use a secure key in production

DATABASE = 'database/users.db'

# --- Helper Functions ---
DATABASE = 'database/users.db'

import sqlite3

DATABASE = 'database/users.db'

def init_db():
    """Initializes the database with users, approval_ids, and inventory tables."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # --- USERS TABLE ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            approval_id TEXT NOT NULL
        )
    ''')

    # --- APPROVAL IDS ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS approval_ids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL
        )
    ''')
    cursor.execute("INSERT OR IGNORE INTO approval_ids (code) VALUES (?)", ('WMT-2025-OWN1',))

    # --- INVENTORY TABLE ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL,
            category TEXT,
            brand TEXT,
            stock_quantity INTEGER,
            availability TEXT,
            price_range TEXT,
            rating INTEGER
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialized with all tables.")




def get_user(username):
    """Fetch user by username."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

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
    result_list = []
    total_sales = 0.0

    for i, week in enumerate(forecast_result, 1):
        entry = {
            "week": i,
            "From": week['From'],
            "Till": week['Till'],
            "sales_prediction": round(week['sales_prediction'], 2)
        }
        if 'note' in week:
            entry['note'] = week['note']

        total_sales += week['sales_prediction']
        result_list.append(entry)

    return {
        "forecast_details": result_list,
        "total_sales": round(total_sales, 2)
    }


def preprocess_walmart_data():
    # Load raw data
    df = pd.read_csv("Walmart.csv")

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    # Feature Engineering
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Week_of_Month'] = df['Date'].dt.day // 7 + 1

    df['Sales_Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
    df['Sales_Lag_2'] = df.groupby('Store')['Weekly_Sales'].shift(2)
    df['Rolling_Mean'] = df.groupby('Store')['Weekly_Sales'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

    df['Sales_Lag_1'] = df.groupby('Store')['Sales_Lag_1'].transform(lambda x: x.fillna(x.mean()))
    df['Sales_Lag_2'] = df.groupby('Store')['Sales_Lag_2'].transform(lambda x: x.fillna(x.mean()))
    df['Rolling_Mean'] = df.groupby('Store')['Rolling_Mean'].transform(lambda x: x.fillna(x.mean()))

    # Encoding store
    le_store = LabelEncoder()
    df['Store_Encoded'] = le_store.fit_transform(df['Store'])

    # Scaling
    df['Temperature_Scaled'] = StandardScaler().fit_transform(df[['Temperature']])
    df['Fuel_Price_Scaled'] = StandardScaler().fit_transform(df[['Fuel_Price']])
    df['CPI_Scaled'] = StandardScaler().fit_transform(df[['CPI']])
    df['Unemployment_Scaled'] = StandardScaler().fit_transform(df[['Unemployment']])
    df['Sales_Lag_1_Scaled'] = StandardScaler().fit_transform(df[['Sales_Lag_1']])
    df['Sales_Lag_2_Scaled'] = StandardScaler().fit_transform(df[['Sales_Lag_2']])
    df['Rolling_Mean_Scaled'] = StandardScaler().fit_transform(df[['Rolling_Mean']])

    df['Holiday_CPI_Scaled'] = df['Holiday_Flag'] * df['CPI_Scaled']
    df['Holiday_Temperature_Scaled'] = df['Holiday_Flag'] * df['Temperature_Scaled']

    df['Specific_Holiday_Encoded'] = df['Holiday_Flag']  # Placeholder for actual holiday types

    df.to_csv("walmart_preprocessed_enhanced_v2.csv", index=False)
    return df


def compute_store_metrics():
    df = pd.read_csv('walmart_preprocessed_enhanced_v2.csv', parse_dates=['Date'])
    result_list = []

    for store_id in sorted(df['Store'].unique()):
        df_store = df[df['Store'] == store_id]

        total_sales = df_store['Weekly_Sales'].sum()
        avg_sales = df_store['Weekly_Sales'].mean()
        max_sales = df_store['Weekly_Sales'].max()
        min_sales = df_store['Weekly_Sales'].min()

        avg_cpi = df_store['CPI'].mean()
        avg_unemployment = df_store['Unemployment'].mean()

        total_weeks = df_store['Week'].nunique()
        holiday_weeks = df_store[df_store['Holiday_Flag'] == 1]['Week'].nunique()
        non_holiday_weeks = total_weeks - holiday_weeks

        start_date = df_store['Date'].min()
        end_date = df_store['Date'].max()
        year_range = df_store['Year'].nunique()

        result = {
            "store_id": int(store_id),
            "sales_based": {
                "total_sales": round(total_sales, 2),
                "average_weekly_sales": round(avg_sales, 2),
                "max_weekly_sales": round(max_sales, 2),
                "min_weekly_sales": round(min_sales, 2)
            },
            "economic_indicators": {
                "avg_cpi": round(avg_cpi, 2),
                "avg_unemployment": round(avg_unemployment, 2)
            },
            "operational_metrics": {
                "total_weeks": total_weeks,
                "holiday_weeks": holiday_weeks,
                "non_holiday_weeks": non_holiday_weeks
            },
            "time_based": {
                "start": start_date.strftime('%Y-%m-%d'),
                "end": end_date.strftime('%Y-%m-%d'),
                "year_range": year_range
            }
        }

        result_list.append(result)

    result_list.sort(key=lambda x: x['sales_based']['total_sales'], reverse=True)
    return result_list


def get_all_store_metrics():
    # Load preprocessed data
    df = pd.read_csv('walmart_preprocessed_enhanced_v2.csv', parse_dates=['Date'])

    result_list = []

    # Loop through each unique store
    for store_id in sorted(df['Store'].unique()):
        df_store = df[df['Store'] == store_id]

        # --- Sales-Based Metrics ---
        total_sales = df_store['Weekly_Sales'].sum()
        avg_sales = df_store['Weekly_Sales'].mean()
        max_sales = df_store['Weekly_Sales'].max()
        min_sales = df_store['Weekly_Sales'].min()

        # --- Economic Indicators ---
        avg_cpi = df_store['CPI'].mean()
        avg_unemployment = df_store['Unemployment'].mean()

        # --- Operational Metrics ---
        total_weeks = df_store['Week'].nunique()
        holiday_weeks = df_store[df_store['Holiday_Flag'] == 1]['Week'].nunique()
        non_holiday_weeks = total_weeks - holiday_weeks

        # --- Time-Based Info ---
        start_date = df_store['Date'].min()
        end_date = df_store['Date'].max()
        year_range = df_store['Year'].nunique()

        result = {
            "store_id": int(store_id),
            "sales_based": {
                "total_sales": round(total_sales, 2),
                "average_weekly_sales": round(avg_sales, 2),
                "max_weekly_sales": round(max_sales, 2),
                "min_weekly_sales": round(min_sales, 2)
            },
            "economic_indicators": {
                "avg_cpi": round(avg_cpi, 2),
                "avg_unemployment": round(avg_unemployment, 2)
            },
            "operational_metrics": {
                "total_weeks": total_weeks,
                "holiday_weeks": holiday_weeks,
                "non_holiday_weeks": non_holiday_weeks
            },
            "time_based": {
                "start": start_date.strftime('%Y-%m-%d'),
                "end": end_date.strftime('%Y-%m-%d'),
                "year_range": year_range
            }
        }

        result_list.append(result)

    # Sort all stores by total_sales descending
    result_list.sort(key=lambda x: x['sales_based']['total_sales'], reverse=True)

    return result_list


def generate_store_sales_plot(store_id, input_csv='walmart_preprocessed_enhanced_v2.csv'):
    try:
        df = pd.read_csv(input_csv)
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter for the selected store
        store_df = df[df['Store'] == store_id]
        if store_df.empty:
            return f"<p>No data available for Store {store_id}.</p>"

        # Group by month
        store_df['Month'] = store_df['Date'].dt.to_period('M').dt.to_timestamp()
        monthly_sales = store_df.groupby('Month')['Weekly_Sales'].sum().reset_index()

        # Plot
        fig = px.line(
            monthly_sales,
            x='Month_Year',
            y='Weekly_Sales',
            title=f'Monthly Sales for Store {store_id}',
            labels={'Month_Year': 'Month-Year', 'Weekly_Sales': 'Total Sales'}
        )

        fig.update_layout(
            template='plotly_white',
            height=500,
            autosize=True,
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis=dict(
                title='Month-Year',
                tickangle=45,
                showgrid=True,
                zeroline=True,
                linecolor='black',
                mirror=True
            ),
            yaxis=dict(
                title='Total Sales',
                showgrid=True,
                zeroline=True,
                linecolor='black',
                mirror=True
            )
        )

        return pio.to_html(fig, full_html=False)

    except Exception as e:
        return f"<p> Error generating plot: {e}</p>"
    

def get_critical_alerts():
    conn = sqlite3.connect('database/users.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM inventory")
    rows = cursor.fetchall()

    alerts = []
    for row in rows:
        stock_quantity = row["stock_quantity"]
        percent = (stock_quantity / 200) * 100
        if percent <= 30:
            alerts.append(f"🚨 {row['item_name']} is in critical stock ({stock_quantity} units left)")
    return alerts



# --- Routes ---

@app.context_processor
def inject_alerts():
    alerts = get_critical_alerts()
    return dict(alerts=alerts)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        store_id = data['store_id']
        start = data['start']
        end = data['end']

        # Preprocess full dataset
        df_processed = preprocess_walmart_data()

        # Forecast using the processed dataframe
        forecast_result = forecast_weekly_sales(start, end, store_id)

        # Format results
        forecast_debug = debug_forecast_output(forecast_result)

        total_predicted_sales = forecast_debug['total_sales']
        total_days = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
        daily_avg = total_predicted_sales / total_days if total_days > 0 else 0

        return jsonify({
            "store": store_id,
            "start": start,
            "end": end,
            "days": total_days,
            "predicted_sales": round(total_predicted_sales, 2),
            "daily_average": round(daily_avg, 2),
            "forecast_details": forecast_debug["forecast_details"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/top-store-metrics')
def top_store_metrics():
    df = pd.read_csv('walmart_preprocessed_enhanced_v2.csv', parse_dates=['Date'])

    result_list = []

    for store_id in sorted(df['Store'].unique()):
        df_store = df[df['Store'] == store_id]

        result = {
            "store_id": int(store_id),
            "sales_based": {
                "total_sales": round(df_store['Weekly_Sales'].sum(), 2),
                "average_weekly_sales": round(df_store['Weekly_Sales'].mean(), 2),
                "max_weekly_sales": round(df_store['Weekly_Sales'].max(), 2),
                "min_weekly_sales": round(df_store['Weekly_Sales'].min(), 2)
            },
            "economic_indicators": {
                "avg_cpi": round(df_store['CPI'].mean(), 2),
                "avg_unemployment": round(df_store['Unemployment'].mean(), 2)
            },
            "operational_metrics": {
                "total_weeks": df_store['Week'].nunique(),
                "holiday_weeks": df_store[df_store['Holiday_Flag'] == 1]['Week'].nunique(),
                "non_holiday_weeks": df_store['Week'].nunique() - df_store[df_store['Holiday_Flag'] == 1]['Week'].nunique()
            },
            "time_based": {
                "start": df_store['Date'].min().strftime('%Y-%m-%d'),
                "end": df_store['Date'].max().strftime('%Y-%m-%d'),
                "year_range": df_store['Year'].nunique()
            }
        }

        result_list.append(result)

    result_list.sort(key=lambda x: x['sales_based']['total_sales'], reverse=True)

    return render_template('top.html', metrics=result_list)

@app.route('/')
def root():
    # If user is logged in → go to dashboard
    if 'username' in session:
        return redirect(url_for('dashboard'))

    # Otherwise show login page
    return redirect(url_for('login'))   # FIXED: was 'home' (does NOT exist)


# ================= LOGIN + STATUS (REPLACE THIS BLOCK IN app.py) ==================

# Make sure sessions work
if not getattr(app, 'secret_key', None):
    app.secret_key = "dev_secret_change_me"   # change this in production

from flask import request, render_template, redirect, url_for, session, flash
import sqlite3
from werkzeug.security import check_password_hash

# -------- COMBINED LOGIN ROUTE (GET + POST) --------
@app.route('/login', methods=['GET', 'POST'])
def login():
    # GET → show login page
    if request.method == 'GET':
        return render_template('login.html')

    # POST → authenticate
    form = dict(request.form)
    print("\nDEBUG /login POST:", form)

    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')

    if not username or not password:
        flash("Please enter username and password.")
        return render_template('login.html')

    # Fetch user from DB
    try:
        conn = sqlite3.connect(DATABASE)
        cur = conn.cursor()
        cur.execute("SELECT id, username, password FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        conn.close()
    except Exception as e:
        print("DB ERROR:", e)
        flash("Internal server error.")
        return render_template('login.html')

    print("DEBUG user from DB:", user)

    if not user:
        flash("User not found.")
        return render_template('login.html')

    stored_pw = user[2]

    # Validate password (hashed OR plain)
    valid = False
    try:
        valid = check_password_hash(stored_pw, password)
    except:
        valid = (stored_pw == password)

    if not valid:
        flash("Invalid username or password.")
        return render_template('login.html')

    # SUCCESS → store session + redirect
    session['username'] = username
    print(f"LOGIN SUCCESS for {username} → redirecting to /dashboard\n")
    return redirect(url_for('dashboard'))


# -------- FIXED STATUS ROUTE (BROKEN REDIRECT FIXED) --------
@app.route('/status')
def analytics():
    if 'username' in session:
        return render_template('analytics.html',
                               username=session['username'],
                               plot_html="")
    else:
        return redirect(url_for('login'))   # FIX: was login_page (does NOT exist)

# ==================================================================


@app.route('/index')
def index():
    if 'username' in session:
        username = session['username']
        return render_template('index.html', username=username)
    return redirect(url_for('login'))



def is_valid_approval_id(approval_id):
    """Check if the entered approval ID exists in the approval_ids table."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM approval_ids WHERE code = ?", (approval_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        approval_id = request.form['approval_id']

        # Validation checks
        if password != confirm_password:
            return render_template('signup.html', error='Passwords do not match.')

        if get_user(username):
            return render_template('signup.html', error='Username already exists.')

        if not is_valid_approval_id(approval_id):
            return render_template('signup.html', error='Invalid Approval ID.')

        # Hash the password and save to DB
        password_hash = generate_password_hash(password)
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password, approval_id) VALUES (?, ?, ?)",
                       (username, password_hash, approval_id))
        conn.commit()
        conn.close()

        # Auto-login by setting session
        session['username'] = username

        # Redirect to dashboard
        return redirect(url_for('dashboard'))

    return render_template('signup.html')



@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    else:
        return redirect(url_for('home'))




@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))



@app.route('/addstock', methods=['GET', 'POST'])
def add_stock():
    if request.method == 'POST':
        item_name = request.form.get('item_name')
        category = request.form.get('category')
        brand = request.form.get('brand')
        stock_quantity = request.form.get('stock_quantity')
        availability = request.form.get('availability')
        price_range = request.form.get('price_range')
        rating = request.form.get('rating')

        # ✅ Save to database
        conn = sqlite3.connect('database/users.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO inventory 
            (item_name, category, brand, stock_quantity, availability, price_range, rating)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (item_name, category, brand, stock_quantity, availability, price_range, rating))
        conn.commit()
        conn.close()

        return render_template('addstock.html', message="✅ Item added successfully!")

    return render_template('addstock.html')


@app.route('/stockstatus')
def stock_status():
    conn = sqlite3.connect('database/users.db')  # ✅ use your actual DB file
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM inventory")
    rows = cursor.fetchall()

    categories = list({row["category"] for row in rows})
    brands = list({row["brand"] for row in rows})

    alerts = []
    for row in rows:
        stock_quantity = row["stock_quantity"]
        percent = (stock_quantity / 200) * 100
        if percent <= 30:
            alerts.append(f"🚨 {row['item_name']} is in critical stock ({stock_quantity} units left)")

    return render_template("status.html",
                           products=rows,
                           categories=categories,
                           brands=brands,
                           alerts=alerts)  # 🔁 include alerts in context


@app.route('/billing', methods=['GET', 'POST'])
def billing():
    message = None
    success = False

    if request.method == 'POST':
        product_id = request.form.get('product_id')
        quantity = int(request.form.get('quantity'))

        conn = sqlite3.connect('database/users.db')
        cursor = conn.cursor()

        # Search by product ID or name
        try:
            product_id_int = int(product_id)
        except ValueError:
            product_id_int = -1  # ensures this doesn't match any real id

        cursor.execute("SELECT * FROM inventory WHERE id=? OR item_name=?", (product_id_int, product_id))

        product = cursor.fetchone()

        if product:
            available = int(product[4])  # assuming stock_quantity is at index 4
            if available >= quantity:
                new_qty = available - quantity
                cursor.execute("UPDATE inventory SET stock_quantity=? WHERE id=?", (new_qty, product[0]))
                conn.commit()
                message = f"✅ Billed {quantity} × {product[1]} successfully."
                success = True
            else:
                message = f"❌ Not enough stock. Only {available} left."
        else:
            message = "❌ Product not found."

        conn.close()

    return render_template("billing.html", message=message, success=success)


import plotly.express as px  # Make sure this import is present

@app.route('/store_sales', methods=['POST'])
def store_sales():
    store_id = request.form['store_id']
    df = pd.read_csv('walmart_preprocessed_enhanced_v2.csv')
    store_data = df[df['Store'] == int(store_id)]

    if store_data.empty:
        plot_html = "<p class='text-red-500 text-center'>No data available for this Store ID.</p>"
    else:
        store_data['Month_Year'] = pd.to_datetime(store_data['Date']).dt.to_period('M').astype(str)
        monthly_sales = store_data.groupby('Month_Year')['Weekly_Sales'].sum().reset_index()

        # ✅ Define the line chart
        fig = px.line(
            monthly_sales,
            x='Month_Year',
            y='Weekly_Sales',
            title=f'Monthly Sales for Store {store_id}',
            labels={'Month_Year': 'Month-Year', 'Weekly_Sales': 'Total Sales'}
        )

        # ✅ Make it responsive and broader
        fig.update_layout(
            template='plotly_white',
            autosize=True,
            height=500,
            width=None,  # 👈 ensure width isn't locked
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis=dict(
                title='Month-Year',
                tickangle=45,
                showgrid=True,
                zeroline=True,
                linecolor='black',
                mirror=True
            ),
            yaxis=dict(
                title='Total Sales',
                showgrid=True,
                zeroline=True,
                linecolor='black',
                mirror=True
            )
        )


        # ✅ Embed as responsive HTML
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})

    return render_template('analytics.html', plot_html=plot_html, username=session['username'])


# --- Main ---

if __name__ == '__main__':
    os.makedirs('database', exist_ok=True)
    init_db()
    app.run(debug=True)
