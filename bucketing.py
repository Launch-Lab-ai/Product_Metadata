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
clothing_keywords = [
    'shirt', 'dress', 'apparel', 'footwear', 'saree', 'kurta', 'kurti',
    'tunic', 'anarkali', 'shoes', 'stylish', 'ethnic', 'trendy',
    'fashionwear', 'floral print', 'style code'
]

jewelry_keywords = [
    'wristwatch', 'watch', 'ring', 'diamond', 'gold', 'silver',
    'jewellery', 'jewelry', 'bracelet', 'necklace', 'cubic zirconia'
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

# === Step 4: Label assignment ===
def assign_final_bucket(row):
    name = str(row['product_name']) if pd.notnull(row['product_name']) else ''
    desc = str(row['description']) if pd.notnull(row['description']) else ''
    cat = str(row['product_category_tree']) if pd.notnull(row['product_category_tree']) else ''
    text = f"{name} {desc} {cat}".lower()

    if any(kw in text for kw in jewelry_keywords):
        return "Jewelry"
    elif any(kw in text for kw in clothing_keywords):
        return "Clothing"
    elif any(kw in text for kw in tech_keywords):
        return "Tech & Gadgets"
    elif any(kw in text for kw in home_keywords):
        return "Home & Decor"
    elif row['retail_price'] < 500 or row['discount_percent'] > 60:
        return "Budget Essentials"
    else:
        return "Uncategorized"

def assign_confident_bucket(row):
    name = str(row['product_name']) if pd.notnull(row['product_name']) else ''
    desc = str(row['description']) if pd.notnull(row['description']) else ''
    cat = str(row['product_category_tree']) if pd.notnull(row['product_category_tree']) else ''
    text = f"{name} {desc} {cat}".lower()

    # Strict match for tech
    if 'electronics' in cat and ('bluetooth' in text or 'headphones' in text) and row['retail_price'] > 1000:
        return 'Tech & Gadgets'

    # Strict match for clothing
    if 'apparel' in cat and any(kw in text for kw in ['kurta', 'saree', 'style code']):
        return 'Clothing'

    # Strict match for home
    if 'decor' in cat or any(kw in text for kw in ['cushion', 'curtain', 'lamp']):
        return 'Home & Decor'

    # Budget: price + weak language
    if row['retail_price'] < 500 and 'affordable' in text:
        return 'Budget Essentials'

    return 'Uncertain'

df['confident_bucket'] = df.apply(assign_confident_bucket, axis=1)
labeled_df = df[df['confident_bucket'] != 'Uncertain']
labeled_df.to_csv('./data/flipkart_labeled_seed.csv', index=False)

df['final_bucket'] = df.apply(assign_final_bucket, axis=1)

# === Step 5: Save labeled dataset ===
df.to_csv('./data/flipkart_buckets_single_label.csv', index=False)
print("✅ Saved final labeled data to './data/flipkart_buckets_single_label.csv'")

# === Step 6: Bucket counts ===
bucket_counts = df['final_bucket'].value_counts()

print("\n📦 Product Counts by Final Bucket:")
for label, count in bucket_counts.items():
    print(f"➡️  {label:18s}: {count}")

# === Step 7: Visualization ===
plt.figure(figsize=(10, 6))
bars = plt.bar(bucket_counts.index, bucket_counts.values, color=['#5DADE2', '#EC7063', '#58D68D', '#A569BD', '#F4D03F', '#95A5A6'])

# Add value labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 100, f'{yval}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("📊 Product Count per Bucket", fontsize=16)
plt.xlabel("Bucket", fontsize=12)
plt.ylabel("Number of Products", fontsize=12)
plt.xticks(rotation=15, fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()
