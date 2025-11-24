import pandas as pd
import numpy as np

# Parameters
n_days = 80
start_date = "2024-01-01"
store_id = "S001"
product_id = "P0001"

# Generate smooth sales trend
np.random.seed(42)
dates = pd.date_range(start=start_date, periods=n_days)
units_sold = np.random.randint(90, 110, size=n_days)  # mostly stable sales

# Inject an anomaly (spike)
units_sold[15] = 200  # day 16 is an anomaly

# Other columns
price = np.random.uniform(10, 15, size=n_days)
discount = np.random.uniform(0, 2, size=n_days)
competitor_pricing = np.random.uniform(10, 15, size=n_days)
inventory_level = np.random.randint(50, 150, size=n_days)

# Create DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Store ID": [store_id]*n_days,
    "Product ID": [product_id]*n_days,
    "Units Sold": units_sold,
    "Price": price,
    "Discount": discount,
    "Competitor Pricing": competitor_pricing,
    "Inventory Level": inventory_level
})

# Save to CSV
file_path = "synthetic_sales_with_anomaly.csv"
df.to_csv(file_path, index=False)
print(f"CSV file created: {file_path}")
