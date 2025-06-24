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

    name = str(row['product_name']) if pd.notnull(row['product_name']) else ''
    desc = str(row['description']) if pd.notnull(row['description']) else ''
    cat = str(row['product_category_tree']) if pd.notnull(row['product_category_tree']) else ''
    text = f"{name} {desc} {cat}".lower()

    if row['retail_price'] < 500 or row['discount_percent'] > 60:
        buckets.append("Budget")

    tech_keywords = ['smart', 'usb', 'bluetooth', 'led', 'electronics',
                     'sound mixer', 'equalizer', 'dj', 'digital', 'display',
                     'chronograph', 'alarm', 'backlight', 'stopwatch', 'dual time', 'tech', 'audio']
    if contains_keywords(text, tech_keywords):
        buckets.append("Tech")

    minimal_keywords = ['minimal', 'simple', 'plain', 'white', 'matte', 'clean', 'compact']
    if contains_keywords(text, minimal_keywords):
        buckets.append("Minimalist")

    fashion_keywords = [
        'shirt', 'dress', 'apparel', 'footwear', 'fashion', 'saree',
        'clothing', 'shoes', 'kurta', 'kurti', 'tunic', 'anarkali',
        'cotton', 'floral', 'pattern', 'style code', 'watch', 'analog',
        'wristwatch', 'ring', 'diamond', 'gold', 'silver', 'jewellery',
        'jewelry', 'amethyst', 'ruby', 'emerald', 'bracelet', 'cubic zirconia'
    ]
    if contains_keywords(text, fashion_keywords):
        buckets.append("Fashion")

    home_keywords = ['home', 'kitchen', 'decor', 'wall', 'organizer', 'shelf', 'table',
                     'towel', 'bath towel', 'cotton towel', 'vacuum bottle', 'coin bank',
                     'storage', 'bottle', 'container', 'utility']
    if contains_keywords(text, home_keywords):
        buckets.append("Home & Decor")

    sports_keywords = ['sport', 'skating', 'fitness', 'exercise', 'skates', 'helmet']
    if contains_keywords(text, sports_keywords):
        buckets.append("Sports & Fitness")

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

# plt.figure(figsize=(10, 5))
# plt.bar(desired_order, counts_ordered, color='skyblue')
# plt.title('📊 Product Count per Bucket')
# plt.xlabel('Bucket')
# plt.ylabel('Number of Products')
# plt.xticks(rotation=20)
# plt.tight_layout()
# plt.show()


# === Step 7: Print individual bucket counts ===
print("\n📦 Product Counts by Bucket:")
for label, count in zip(desired_order, counts_ordered):
    print(f"➡️  {label:15s}: {count}")



print(df['buckets'].explode().value_counts())


# Filter for Uncategorized products
uncategorized_df = df[df['buckets'].apply(lambda x: 'Uncategorized' in x)]

# Show product name + description for top 10
print("\n🔎 Sample 'Uncategorized' Products (Name + Description):\n")
for i, row in uncategorized_df.head(20).iterrows():
    print(f"🟦 Product {i+1}")
    print(f"🔹 Name       : {row['product_name']}")
    print(f"📝 Description: {row['description']}\n")
