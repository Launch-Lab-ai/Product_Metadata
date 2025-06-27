import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

def main():
    # === Step 1: Load model and vectorizer (AMAZON versions) ===
    model_path = './models/amazon/bucket_classifier.pkl'
    vectorizer_path = './models/amazon/vectorizer.pkl'
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # === Step 2: Prepare for batch processing ===
    data_path = './data/amazon/amazon_cleaned.csv'
    output_path = './data/amazon/amazon_predicted_buckets.csv'
    chunksize = 100000
    first_write = True
    bucket_counts = {}

    # ✅ Fix: Use UTF-8 to avoid UnicodeDecodeError
    with open(data_path, encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  # exclude header

    total_chunks = (total_rows // chunksize) + int(total_rows % chunksize > 0)
    print(f"\n📦 Total chunks to process: {total_chunks}\n")

    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunksize, encoding='utf-8')):
        chunk['text'] = chunk['TITLE'].fillna('') + ' ' + chunk['DESCRIPTION'].fillna('') + ' ' + chunk['BULLET_POINTS'].fillna('')
        X_vec = vectorizer.transform(chunk['text'])
        chunk['predicted_bucket'] = clf.predict(X_vec)

        # Count predictions
        counts = chunk['predicted_bucket'].value_counts().to_dict()
        for label, count in counts.items():
            bucket_counts[label] = bucket_counts.get(label, 0) + count

        # Save chunk predictions
        chunk.to_csv(output_path, mode='w' if first_write else 'a', header=first_write, index=False)
        first_write = False
        print(f"✅ Processed chunk {i+1}/{total_chunks} — predictions saved")

    # === Step 3: Print bucket summary in terminal ===
    print("\n📦 Final Predicted Bucket Counts:")
    for label, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        print(f"➡️  {label:25s}: {count}")

    # === Step 4: Plot predicted bucket distribution ===
    plt.figure(figsize=(12, 7))
    sorted_counts = dict(sorted(bucket_counts.items(), key=lambda x: -x[1]))
    bars = plt.bar(sorted_counts.keys(), sorted_counts.values(), color=plt.cm.tab20.colors)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + max(sorted_counts.values())*0.01, f'{yval}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title("ML-Predicted Amazon Product Bucket Distribution", fontsize=18)
    plt.xlabel("Buckets", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Products", fontsize=14, fontweight='bold')
    plt.xticks(rotation=25, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    fig_dir = './figures/amazon'
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'bucket_prediction.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n🖼️ Saved bucket distribution chart to '{fig_path}'")
    plt.show()

if __name__ == '__main__':
    main()
