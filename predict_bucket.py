import pandas as pd
import joblib
import matplotlib.pyplot as plt
from collections import Counter
import os

# === Step 1: Load model and vectorizer ===
model = joblib.load('./models/bucket_classifier.pkl')
vectorizer = joblib.load('./models/vectorizer.pkl')

# === Step 2: Load full cleaned data ===
df = pd.read_csv('./data/flipkart_cleaned.csv')

# === Step 3: Combine text fields for prediction ===
df['text'] = (df['product_name'].fillna('') + ' ' +
              df['description'].fillna('') + ' ' +
              df['product_category_tree'].fillna(''))

# === Step 4: Vectorize and predict ===
X_full = vectorizer.transform(df['text'])
df['predicted_bucket'] = model.predict(X_full)

# === Step 5: Save predictions ===
os.makedirs('./data', exist_ok=True)
df.to_csv('./data/predicted_buckets.csv', index=False)
print("✅ Predictions saved to './data/predicted_buckets.csv'")

# === Step 6: Bucket counts ===
bucket_counts = df['predicted_bucket'].value_counts().sort_values(ascending=False)

print("\n📦 Final Predicted Bucket Counts:\n")
for label, count in bucket_counts.items():
    print(f"➡️  {label:18s}: {count}")

# === Step 7: Plot and save bar chart ===
plt.figure(figsize=(10, 6))
bars = plt.bar(bucket_counts.index, bucket_counts.values, color=['#5DADE2', '#EC7063', '#58D68D', '#A569BD', '#F4D03F', '#95A5A6'])

# Add value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 100, f'{yval}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Chart style
plt.title("ML-Predicted Product Bucket Distribution", fontsize=16)
plt.xlabel("Buckets", fontsize=12, fontweight='bold')
plt.ylabel("Number of Products", fontsize=12, fontweight='bold')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Save and show
os.makedirs('./figures', exist_ok=True)
plt.savefig('./figures/bucket_prediction.png', dpi=300, bbox_inches='tight')
print("🖼️ Saved bar chart to './figures/bucket_prediction.png'")
plt.show()
