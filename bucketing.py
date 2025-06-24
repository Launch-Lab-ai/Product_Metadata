import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# === Step 1: Load cleaned data ===
df = pd.read_csv('./data/flipkart_cleaned.csv')

# === Step 2: Add discount percent ===
def calculate_discount(row):
    if row['retail_price'] > 0:
        return ((row['retail_price'] - row['discounted_price']) / row['retail_price']) * 100
    return 0

df['discount_percent'] = df.apply(calculate_discount, axis=1)

# === Step 3: Define bucket logic ===
def contains_keywords(text, keywords):
    return any(kw in text for kw in keywords)

def assign_buckets(row):
    buckets = []

    # Safe string conversion
    name = str(row['product_name']) if pd.notnull(row['product_name']) else ''
    desc = str(row['description']) if pd.notnull(row['description']) else ''
    cat = str(row['product_category_tree']) if pd.notnull(row['product_category_tree']) else ''
    text = name + " " + desc + " " + cat

    # 1. Budget
    if row['retail_price'] < 500 or row['discount_percent'] > 60:
        buckets.append("Budget")

    # 2. Tech
    tech_keywords = ['smart', 'usb', 'bluetooth', 'led', 'electronics']
    if contains_keywords(text, tech_keywords):
        buckets.append("Tech")

    # 3. Minimalist
    minimal_keywords = ['minimal', 'simple', 'plain', 'white', 'matte', 'clean', 'compact']
    if contains_keywords(text, minimal_keywords):
        buckets.append("Minimalist")

    # 4. Fashion
    fashion_keywords = ['shirt', 'dress', 'apparel', 'footwear', 'fashion', 'saree', 'clothing', 'shoes']
    if contains_keywords(text, fashion_keywords):
        buckets.append("Fashion")

    # 5. Home & Decor
    home_keywords = ['home', 'kitchen', 'decor', 'wall', 'organizer', 'shelf', 'table']
    if contains_keywords(text, home_keywords):
        buckets.append("Home & Decor")

    return buckets if buckets else ["Uncategorized"]

# Apply bucketing
df['buckets'] = df.apply(assign_buckets, axis=1)

# === Step 4: Count individual bucket frequencies ===
flat_list = [bucket for sublist in df['buckets'] for bucket in sublist]
bucket_counts = Counter(flat_list)

# === Step 5: Save final data with buckets ===
df['bucket_label'] = df['buckets'].apply(lambda x: ', '.join(x))
df.to_csv('./data/flipkart_buckets.csv', index=False)
print("✅ Saved bucketed data to './data/flipkart_buckets.csv'")

# === Step 6: Plot bar chart for exactly 6 categories ===
desired_order = ['Budget', 'Tech', 'Minimalist', 'Fashion', 'Home & Decor', 'Uncategorized']
counts_ordered = [bucket_counts.get(label, 0) for label in desired_order]

plt.figure(figsize=(10, 5))
plt.bar(desired_order, counts_ordered, color='skyblue')
plt.title('📊 Product Count per Bucket')
plt.xlabel('Bucket')
plt.ylabel('Number of Products')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
