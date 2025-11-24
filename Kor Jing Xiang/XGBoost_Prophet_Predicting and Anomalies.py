import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load dataset
df = pd.read_csv("retail_store_inventory.csv")

# Convert date and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
STORE_TO_USE = "S001"
PRODUCT_TO_USE = "P0001"

df = df[(df["Store ID"] == STORE_TO_USE) &
        (df["Product ID"] == PRODUCT_TO_USE)]
df["Store_ID_encoded"] = df["Store ID"].astype("category").cat.codes
df["Product_ID_encoded"] = df["Product ID"].astype("category").cat.codes

# Time features
df["day_of_week"] = df["Date"].dt.dayofweek
df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
df["month"] = df["Date"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Price elasticity features
df["price_change"] = df.groupby("Product ID")["Price"].pct_change()
df["discount_percent"] = df["Discount"] / df["Price"]
df["price_comp_ratio"] = df["Price"] / (df["Competitor Pricing"] + 1e-9)
df["price_discount_ratio"] = df["Price"] * df["discount_percent"]

# Target
target = "Units Sold"

# Create lag features
for lag in [1, 2, 3, 7, 14]:
    df[f"units_lag_{lag}"] = df.groupby(["Store ID", "Product ID"])["Units Sold"].shift(lag)
df["diff_1"] = df.groupby(["Store ID", "Product ID"])["Units Sold"].diff(1)
df["diff_7"] = df.groupby(["Store ID", "Product ID"])["Units Sold"].diff(7)

# Rolling window features
df["rolling_mean_7"] = (
    df.groupby(["Store ID","Product ID"])["Units Sold"]
      .shift(1).rolling(7).mean()
)

df["rolling_std_7"] = (
    df.groupby(["Store ID","Product ID"])["Units Sold"]
      .shift(1).rolling(7).std()
)

df["rolling_mean_30"] = (
    df.groupby(["Store ID","Product ID"])["Units Sold"]
      .shift(1).rolling(30).mean()
)

df = df.dropna().reset_index(drop=True)

# Features
feature_cols = [
    'Inventory Level', 'Price', 'Discount', 'Competitor Pricing',
    'day_of_week', 'week_of_year', 'month', 'is_weekend',
]
# Encoded features
feature_cols += ["Store_ID_encoded", "Product_ID_encoded"]

# Price-related features
feature_cols += [
    "price_change", "discount_percent",
    "price_comp_ratio", "price_discount_ratio"
]

# Momentum features
feature_cols += ["diff_1", "diff_7"]


feature_cols += [f"units_lag_{lag}" for lag in [1,2,3,7,14]]
feature_cols += ["rolling_mean_7", "rolling_std_7", "rolling_mean_30"]

X = df[feature_cols]
y = df[target]

# Train-test split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# XGBoost model
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=12,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=1,
    reg_lambda=2,
    min_child_weight=5,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

file_path = r"C:\Users\KOR JING XIANG\Hackathon\System.out.print-Dora-the-explorer-\synthetic_sales_with_anomaly.csv"
print("Using file:", file_path)
user_df = pd.read_csv(file_path)
user_df["Date"] = pd.to_datetime(user_df["Date"])

""" #syntax for user to upload csv, change back if works
Tk().withdraw()  # Hide root window
file_path = askopenfilename(
    title="Select CSV file with actual sales",
    filetypes=[("CSV Files", "*.csv")]
)

if not file_path:
    raise SystemExit("No file selected. Exiting...")
"""
#-----------------------------------------------
# ASK USER HOW MANY DAYS TO FORECAST
#-----------------------------------------------
days = int(input("How many future days to forecast? "))
# ----------------------------------------------
# SELECT WHICH PRODUCT + STORE TO FORECAST
# ----------------------------------------------
PRODUCT_TO_FORECAST = "P0001"
STORE_TO_FORECAST = "S001"

# ----------------------------------------------
# Compute historical rolling threshold
# ----------------------------------------------
user_df_target = user_df[(user_df["Product ID"] == PRODUCT_TO_FORECAST) &
                         (user_df["Store ID"] == STORE_TO_FORECAST)].copy()

rolling_mean = user_df_target['Units Sold'].rolling(7, min_periods=1).mean()
rolling_std  = user_df_target['Units Sold'].rolling(7, min_periods=1).std()
dynamic_threshold = rolling_mean + 2 * rolling_std
threshold_for_forecast = dynamic_threshold.iloc[-1]  # last available value


from prophet import Prophet

def forecast_future_days_prophet(df, product_id, store_id, n_days=365):
    """
    Forecast using Prophet for long-term predictions.
    """
    # Filter target product + store
    df_target = df[(df["Product ID"] == product_id) & 
                   (df["Store ID"] == store_id)][["Date", "Units Sold"]].copy()
    
    df_target = df_target.rename(columns={"Date": "ds", "Units Sold": "y"})
    
    # Initialize Prophet model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_target)
    
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=n_days)
    
    # Predict
    forecast = model.predict(future)
    
    # Only return the future predictions
    result = forecast[["ds", "yhat"]].tail(n_days).rename(columns={"ds": "Date", "yhat": "Predicted Units Sold"})
    return result


def forecast_future_days(df, model, feature_cols, product_id, store_id, n_days=365):
    """
    Forecast future units sold for a given product and store.
    Handles rolling windows and lag features recursively for long horizons.
    """

    # Filter target product + store
    df_target = df[(df["Product ID"] == product_id) &
                   (df["Store ID"] == store_id)].copy()
    df_target = df_target.sort_values("Date").reset_index(drop=True)

    # Initialize lag history
    max_lag = 14
    for lag in range(1, max_lag + 1):
        if f"units_lag_{lag}" not in df_target.columns:
            df_target[f"units_lag_{lag}"] = df_target["Units Sold"].shift(lag)

    # Start from last row
    current_row = df_target.iloc[-1].copy()
    future_rows = []

    # Maintain rolling windows
    rolling7 = list(df_target["Units Sold"].iloc[-7:]) if len(df_target) >= 7 else list(df_target["Units Sold"])
    rolling30 = list(df_target["Units Sold"].iloc[-30:]) if len(df_target) >= 30 else list(df_target["Units Sold"])

    for _ in range(n_days):
        next_date = current_row["Date"] + pd.Timedelta(days=1)
        future = current_row.copy()
        future["Date"] = next_date

        # --- Time features ---
        future["day_of_week"] = next_date.dayofweek
        future["week_of_year"] = next_date.isocalendar().week
        future["month"] = next_date.month
        future["is_weekend"] = int(next_date.dayofweek >= 5)

        # --- Update lag features ---
        future["units_lag_1"] = current_row["Units Sold"]
        for i in range(2, max_lag + 1):
            future[f"units_lag_{i}"] = current_row.get(f"units_lag_{i-1}", 0)

        # --- Update rolling windows with predicted values ---
        rolling7.append(future["units_lag_1"])
        rolling30.append(future["units_lag_1"])
        rolling7 = rolling7[-7:]
        rolling30 = rolling30[-30:]

        future["rolling_mean_7"] = sum(rolling7) / len(rolling7)
        future["rolling_std_7"] = pd.Series(rolling7).std()
        future["rolling_mean_30"] = sum(rolling30) / len(rolling30)

        # --- Price features ---
        future["discount_percent"] = future["Discount"] / future["Price"]
        future["price_change"] = (future["Price"] - current_row["Price"]) / current_row["Price"]
        future["price_comp_ratio"] = future["Price"] / (future["Competitor Pricing"] + 1e-9)
        future["price_discount_ratio"] = future["Price"] * future["discount_percent"]

        # --- Predict ---
        pred = model.predict(future[feature_cols].values.reshape(1, -1))[0]
        future_rows.append([next_date, pred])

        # --- Update current_row for next iteration ---
        current_row["Date"] = next_date
        current_row["Units Sold"] = pred
        for i in range(1, max_lag + 1):
            current_row[f"units_lag_{i}"] = future[f"units_lag_{i}"]

    return pd.DataFrame(future_rows, columns=["Date", "Predicted Units Sold"])

#detect and flag future risks
def detect_future_risks(forecast_df, threshold):
    """
    Flag potential future stock risks based on forecasted units sold.
    threshold: scalar value from historical rolling mean + 2*rolling std
    """
    df = forecast_df.copy()

    # Flag risk if predicted units exceed dynamic threshold
    df['Risk'] = df['Predicted Units Sold'] > threshold

    # Add message explaining the risk
    df['Message'] = df.apply(
        lambda row: f"Predicted demand ({row['Predicted Units Sold']:.0f}) exceeds expected threshold ({threshold:.0f})"
        if row['Risk'] else "",
        axis=1
    )

    return df[['Date', 'Predicted Units Sold', 'Risk', 'Message']]

# ----------------------------------------------
# GENERATE FORECAST
# ----------------------------------------------
if days < 15:
    # Short-term: XGBoost recursive forecast
    forecast_result = forecast_future_days(
        user_df,
        model,
        feature_cols,
        PRODUCT_TO_FORECAST,
        STORE_TO_FORECAST,
        n_days=days
    )
else:
    # Long-term: Prophet forecast
    forecast_result = forecast_future_days_prophet(
        user_df,
        PRODUCT_TO_FORECAST,
        STORE_TO_FORECAST,
        n_days=days
    )

print(f"\n===== {days}-DAY FORECAST =====")
print(forecast_result)

# ----------------------------------------------
# ANOMALY DETECTION
# ----------------------------------------------

future_risks = detect_future_risks(forecast_result, threshold=threshold_for_forecast)
print(future_risks[future_risks['Risk']])
