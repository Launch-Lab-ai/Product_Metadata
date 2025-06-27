import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import time


def main():
    # === Step 1: Load labeled Amazon bucket data in chunks ===
    data_path = './data/amazon/amazon_buckets.csv'
    chunksize = 150000
    chunks = []
    total_chunks = sum(1 for _ in pd.read_csv(data_path, chunksize=chunksize))

    print(f"\U0001F4E6 Total chunks to load: {total_chunks}\n")

    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunksize)):
        print(f"✅ Loaded chunk {i + 1}/{total_chunks}")

        if 'final_bucket' not in chunk.columns or chunk['final_bucket'].isnull().all():
            continue

        chunk = chunk[chunk['final_bucket'] != 'Uncategorized'].copy()
        chunk.loc[:, 'text'] = (
            chunk['TITLE'].fillna('') + ' ' +
            chunk['DESCRIPTION'].fillna('') + ' ' +
            chunk['BULLET_POINTS'].fillna('')
        )

        chunks.append(chunk[['text', 'final_bucket']])

    # Combine and optionally sample training data
    df_train = pd.concat(chunks, ignore_index=True)
    if len(df_train) > 200000:
        print(f"📉 Sampling from {len(df_train)} to 200000 rows for faster training")
        df_train = df_train.sample(n=200000, random_state=42)

    X = df_train['text']
    y = df_train['final_bucket']

    # === Step 2: Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # === Step 3: Vectorize text with TF-IDF ===
    print("🔄 Fitting vectorizer...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # === Step 4: Train Logistic Regression ===
    print("⚡ Training Logistic Regression model...")
    clf = LogisticRegression(max_iter=300, n_jobs=-1)

    start = time.time()
    clf.fit(X_train_vec, y_train)
    end = time.time()
    print(f"✅ Training completed in {end - start:.2f} seconds.")

    # === Step 5: Evaluate model ===
    y_pred = clf.predict(X_test_vec)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # === Step 6: Save model and vectorizer ===
    os.makedirs('./models/amazon', exist_ok=True)
    joblib.dump(clf, './models/amazon/bucket_classifier.pkl')
    joblib.dump(vectorizer, './models/amazon/vectorizer.pkl')

    print("\n✅ Model and vectorizer saved to './models/amazon/'")


if __name__ == '__main__':
    main()
