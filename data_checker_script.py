import pandas as pd

# Load the cleaned Flipkart dataset

# df = pd.read_csv('./data/flipkart.csv')
df = pd.read_csv('./data/flipkart_cleaned.csv')

# -----------------------------
# 📐 Basic Shape & Columns
# -----------------------------
print("📊 Shape of dataset:", df.shape)
print("📄 Columns:\n", df.columns.tolist())

# -----------------------------
# 🔍 Data Preview
# -----------------------------
print("\n🔍 Sample Rows:\n", df.head(5))

# -----------------------------
# ❓ Missing Values
# -----------------------------
print("\n🕳️ Missing Values:\n", df.isnull().sum())

# -----------------------------
# 🔢 Data Types
# -----------------------------
print("\n🧬 Data Types:\n", df.dtypes)

# -----------------------------
# 📈 Describe Numerics
# -----------------------------
print("\n📊 Numeric Summary:\n", df.describe())

# -----------------------------
# 🎯 Rating Distribution
# -----------------------------
print("\n🎯 Sample Rating Columns:")
if 'overall_rating' in df.columns:
    print("✅ Unique values in 'overall_rating':", df['overall_rating'].unique())
if 'product_rating' in df.columns:
    print("✅ Unique values in 'product_rating':", df['product_rating'].unique())


# -----------------------------
# 🧪 Additional Data Checks
# -----------------------------
print("\n🧪 Additional Data Quality Checks:")

# 1. 🔁 Duplicate Rows & Products
print("🔁 Total duplicate rows:", df.duplicated().sum())
if 'product_url' in df.columns:
    print("🔗 Duplicate product URLs:", df['product_url'].duplicated().sum())
if 'product_name' in df.columns:
    print("🛍️ Duplicate product names:", df['product_name'].duplicated().sum())

# 2. ❌ Invalid Rating Values
for col in ['product_rating', 'overall_rating']:
    if col in df.columns:
        invalid_ratings = df[col][~df[col].astype(str).str.replace('.', '', 1).str.isdigit()]
        print(f"⚠️ Invalid entries in '{col}':", invalid_ratings.unique())

# 3. 💸 Discount Validation
if 'retail_price' in df.columns and 'discounted_price' in df.columns:
    wrong_discounts = df[df['discounted_price'] > df['retail_price']]
    print("💸 Products with discounted price > retail price:", wrong_discounts.shape[0])

# 4. 🚩 Extreme Price Outliers (top 1%)
if 'retail_price' in df.columns:
    threshold = df['retail_price'].quantile(0.99)
    print(f"🚩 Products above 99th percentile price ({threshold}):", df[df['retail_price'] > threshold].shape[0])

# 5. 🧪 Product Specifications Format Check
if 'product_specifications' in df.columns:
    spec_errors = df['product_specifications'].apply(lambda x: not isinstance(x, str) or not x.startswith('{'))
    print("🧪 Possibly invalid 'product_specifications' entries:", spec_errors.sum())


# # Show first 10 product_specifications raw values
# for i, val in enumerate(df['product_specifications'].dropna().head(10)):
#     print(f"\n🔢 Row {i}:\n{val}")

