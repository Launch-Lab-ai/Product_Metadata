import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('./data/flipkart.csv')

# Replace "No rating available" with NaN
df['product_rating'] = df['product_rating'].replace("No rating available", np.nan)
df['overall_rating'] = df['overall_rating'].replace("No rating available", np.nan)

# Convert ratings to numeric
df['product_rating'] = pd.to_numeric(df['product_rating'], errors='coerce')
df['overall_rating'] = pd.to_numeric(df['overall_rating'], errors='coerce')

# Fill missing brand names
df['brand'] = df['brand'].fillna('Unknown')

# Convert text fields to lowercase
text_cols = ['product_name', 'description', 'product_category_tree']
for col in text_cols:
    df[col] = df[col].astype(str).str.lower()

# Handle price-related NaNs
df['retail_price'] = df['retail_price'].fillna(0)
df['discounted_price'] = df['discounted_price'].fillna(df['retail_price'])

# Save cleaned data
df.to_csv('./data/flipkart_cleaned.csv', index=False)
print("✅ Cleaned data saved as 'flipkart_cleaned.csv'")
