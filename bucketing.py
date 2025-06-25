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
    'shirt', 'dress', 'apparel', 'saree', 'kurta', 'kurti', 'tunic',
    'anarkali', 'stylish', 'ethnic', 'trendy', 'fashionwear', 'floral print', 'style code',
    't-shirt', 'blouse', 'leggings', 'jeans', 'jacket', 'sweater',
    'boots', 'casual shoes', 'sandals', 'heels', 'footwear', "boy's", "girl's"
]

jewelry_keywords = [
    'wristwatch', 'watch', 'analog watch', 'digital watch', 'bracelet', 'ring',
    'diamond', 'gold', 'silver', 'jewellery', 'jewelry', 'necklace', 'cubic zirconia',
    'chronograph', 'quartz', 'sapphire crystal'
]

tech_keywords = [
    'usb', 'bluetooth', 'led', 'electronics', 'sound mixer', 'equalizer',
    'dj', 'digital display', 'smartphone', 'tablet', 'headphones', 'earphones',
    'amplifier', 'charger', 'adapter', 'speaker', 'hdmi', 'tech', 'gadget',
    'cctv', 'bnc', 'connector', 'wire', 'cable', 'surveillance'
]

home_keywords = [
    'kitchen', 'decor', 'wall art', 'wall sticker', 'planter',
    'storage box', 'laundry bag', 'bed sheet', 'curtain',
    'cutlery', 'cookware', 'lamp', 'cushion', 'vase', 'photo frame',
    'home furnishing', 'organizer', 'tablecloth', 'sofa', 'carpet', 'mat', 'blanket',
    'notebook', 'diary', 'coin bank', 'stationery'
]

# === Step 4: Label assignment ===
def assign_final_bucket(row):
    name = str(row['product_name']) if pd.notnull(row['product_name']) else ''
    desc = str(row['description']) if pd.notnull(row['description']) else ''
    cat = str(row['product_category_tree']) if pd.notnull(row['product_category_tree']) else ''
    text = f"{name} {desc} {cat}".lower()

    if any(kw in text for kw in jewelry_keywords) or 'wrist watches' in cat.lower():
        return "Jewelry"
    elif any(kw in text for kw in clothing_keywords) or 'clothing' in cat.lower() or 'apparel' in cat.lower() or 'footwear' in cat.lower():
        return "Clothing"
    elif any(kw in text for kw in tech_keywords):
        return "Tech & Gadgets"
    elif any(kw in text for kw in home_keywords) or 'decor' in cat.lower() or 'stationery' in cat.lower():
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

# === Step 5: Apply confident labeling first ===
df['confident_bucket'] = df.apply(assign_confident_bucket, axis=1)
labeled_df = df[df['confident_bucket'] != 'Uncertain']
labeled_df.to_csv('./data/flipkart_labeled_seed.csv', index=False)

# === Step 6: Apply full label logic ===
df['final_bucket'] = df.apply(assign_final_bucket, axis=1)

# === Step 7: Save labeled dataset ===
df.to_csv('./data/flipkart_buckets_single_label.csv', index=False)
print("✅ Saved final labeled data to './data/flipkart_buckets_single_label.csv'")

# === Step 8: Bucket counts ===
bucket_counts = df['final_bucket'].value_counts()

print("\n📦 Product Counts by Final Bucket:")
for label, count in bucket_counts.items():
    print(f"➡️  {label:18s}: {count}")

# === Step 9: Visualization ===
plt.figure(figsize=(10, 6))
bars = plt.bar(bucket_counts.index, bucket_counts.values, color=['#5DADE2', '#EC7063', '#58D68D', '#A569BD', '#F4D03F', '#95A5A6'])

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

# === Step 10: View Sample Uncategorized Products ===
print("\n🔎 Sample 'Uncategorized' Products (Name + Description + Category):\n")

uncategorized_df = df[df['final_bucket'] == 'Uncategorized'].head(20)

for i, row in uncategorized_df.iterrows():
    print(f"🟦 Product {i+1}")
    print(f"🔹 Name       : {row['product_name']}")
    print(f"📝 Description: {row.get('description', 'N/A')}")
    print(f"🏷️  Category  : {row.get('product_category_tree', 'N/A')}")
    print(f"💰 Price      : ₹{row['retail_price']}")
    print("-" * 80)
