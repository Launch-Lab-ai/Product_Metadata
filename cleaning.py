import pandas as pd
import numpy as np

# === Step 1: Load the raw Amazon data ===
df = pd.read_csv('./data/amazon/train.csv')

# === Step 2: Drop rows with missing TITLE (essential for classification) ===
df = df.dropna(subset=['TITLE'])

# === Step 3: Drop unnecessary columns ===
df = df.drop(columns=['PRODUCT_TYPE_ID', 'PRODUCT_LENGTH'], errors='ignore')

# === Step 4: Replace missing BULLET_POINTS and DESCRIPTION with empty strings ===
df['BULLET_POINTS'] = df['BULLET_POINTS'].fillna('')
df['DESCRIPTION'] = df['DESCRIPTION'].fillna('')

# === Step 5: Convert all text columns to lowercase ===
text_cols = ['TITLE', 'BULLET_POINTS', 'DESCRIPTION']
for col in text_cols:
    df[col] = df[col].astype(str).str.lower()

# === Step 6: Identify rows with BOTH empty DESCRIPTION and BULLET_POINTS ===
mask_empty_both = (df['DESCRIPTION'].str.strip() == '') & (df['BULLET_POINTS'].str.strip() == '')
count_empty_both = mask_empty_both.sum()
total_rows = len(df)

print(f"📊 Total Products                  : {total_rows}")
print(f"❌ Rows with NO DESCRIPTION + BULLETS: {count_empty_both}")
print(f"✅ Products with at least one field : {total_rows - count_empty_both}")

# === Step 7: Ask whether to drop rows with both fields empty ===
user_input = input("⚠️  Delete rows missing BOTH DESCRIPTION and BULLET_POINTS? (yes/no): ").strip().lower()

if user_input == 'yes':
    df = df[~mask_empty_both]
    df.to_csv('./data/amazon/amazon_cleaned.csv', index=False)
    print("✅ Saved cleaned data (rows with at least one field) to './data/amazon/amazon_cleaned.csv'")
else:
    df.to_csv('./data/amazon/amazon_cleaned.csv', index=False)
    print("📁 No rows deleted. Full cleaned data saved to './data/amazon/amazon_cleaned.csv'")
