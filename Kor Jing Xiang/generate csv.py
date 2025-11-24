import pandas as pd
import numpy as np

# Parameters
dates = pd.date_range(start="2025-11-01", end="2025-11-30")
store_ids = ["S001"]
product_ids = ["P0001"]

data = []

for date in dates:
    for store in store_ids:
        for product in product_ids:
            units_sold = np.random.randint(20, 100)  # Random sales
            inventory = np.random.randint(50, 200)
            price = np.random.uniform(10, 50)
            discount = np.random.uniform(0, 5)
            competitor_price = np.random.uniform(10, 50)
            data.append([
                date, store, product, units_sold,
                inventory, price, discount, competitor_price
            ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "Date", "Store ID", "Product ID", "Units Sold",
    "Inventory Level", "Price", "Discount", "Competitor Pricing"
])

# Save CSV
df.to_csv("sample_sales_data.csv", index=False)
print("sample_sales_data.csv generated!")
