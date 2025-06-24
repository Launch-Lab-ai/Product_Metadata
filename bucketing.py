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

# === Step 3: Define keyword sets ===
fashion_keywords = [
    'shirt', 'dress', 'apparel', 'footwear', 'saree', 'kurta', 'kurti',
    'tunic', 'anarkali', 'shoes', 'wristwatch', 'ring', 'diamond', 'gold',
    'silver', 'jewellery', 'jewelry', 'bracelet', 'necklace', 'stylish',
    'ethnic', 'trendy', 'fashionwear', 'cubic zirconia', 'floral print',
    'style code'
]

tech_keywords = [
    'usb', 'bluetooth', 'led', 'electronics', 'sound mixer', 'equalizer',
    'dj', 'digital display', 'smartphone', 'tablet', 'headphones', 'earphones',
    'amplifier', 'charger', 'adapter', 'speaker', 'hdmi'
]

home_keywords = [
    'kitchen', 'decor', 'wall art', 'wall sticker', 'planter',
    'storage box', 'laundry bag', 'bed sheet', 'curtain',
    'cutlery', 'cookware', 'lamp', 'cushion', 'vase', 'photo frame',
    'home furnishing', 'organizer', 'tablecloth'
]

# === Step 4: Label assignment (single label based on priority) ===
def assign_final_bucket(row):
    name = str(row['product_name']) if pd.notnull(row['product_name']) else ''
    desc = str(row['description']) if pd.notnull(row['description']) else ''
    cat = str(row['product_category_tree']) if pd.notnull(row['product_category_tree']) else ''
    text = f"{name} {desc} {cat}".lower()

    if any(kw in text for kw in fashion_keywords):
        return "Fashion"
    elif any(kw in text for kw in tech_keywords):
        return "Tech"
    elif any(kw in text for kw in home_keywords):
        return "Home & Decor"
    elif row['retail_price'] < 500 or row['discount_percent'] > 60:
        return "Budget"
    else:
        return "Uncategorized"

df['final_bucket'] = df.apply(assign_final_bucket, axis=1)

# === Step 5: Save labeled dataset ===
df.to_csv('./data/flipkart_buckets_single_label.csv', index=False)
print("✅ Saved final labeled data to './data/flipkart_buckets_single_label.csv'")

# === Step 6: Bucket counts ===
bucket_counts = df['final_bucket'].value_counts()

print("\n📦 Product Counts by Final Bucket:")
for label, count in bucket_counts.items():
    print(f"➡️  {label:15s}: {count}")

# Optional: Bar chart
# plt.figure(figsize=(8, 4))
# plt.bar(bucket_counts.index, bucket_counts.values, color='skyblue')
# plt.title("Final Product Bucket Counts")
# plt.xlabel("Bucket")
# plt.ylabel("Count")
# plt.xticks(rotation=15)
# plt.tight_layout()
# plt.show()


# # Uncomment to show chart
# plt.figure(figsize=(10, 5))
# plt.bar(desired_order, counts_ordered, color='skyblue')
# plt.title('📊 Product Count per Bucket')
# plt.xlabel('Bucket')
# plt.ylabel('Number of Products')
# plt.xticks(rotation=20)
# plt.tight_layout()
# plt.show()

# # === Step 7: Print individual bucket counts ===
# print("\n📦 Product Counts by Bucket:")
# for label, count in zip(desired_order, counts_ordered):
#     print(f"➡️  {label:15s}: {count}")

# print(df['buckets'].explode().value_counts())

# # Filter for Uncategorized products
# uncategorized_df = df[df['buckets'].apply(lambda x: 'Uncategorized' in x)]

# # Show product name + description for top 20
# print("\n🔎 Sample 'Uncategorized' Products (Name + Description):\n")
# for i, row in uncategorized_df.head(20).iterrows():
#     print(f"🟦 Product {i+1}")
#     print(f"🔹 Name       : {row['product_name']}")
#     print(f"📝 Description: {row['description']}\n")
