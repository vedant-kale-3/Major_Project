# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "tasks_dataset.csv"
OUT_DIR = "models"

def train_and_save(data_path=DATA_PATH, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    # Ensure strings and drop missing rows (if any)
    df = df.dropna(subset=['Task', 'Category'])
    X = df['Task'].astype(str)
    y = df['Category'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    # NOTE: if your dataset is small, reduce min_df (e.g. min_df=1 or 2)
    tfidf = TfidfVectorizer(min_df=1, stop_words='english',  ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LinearSVC(C=1.0, class_weight='balanced', max_iter=10000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {acc:.4f}\n")
    print("Classification report:\n", classification_report(y_test, y_pred))

    # Confusion matrix saved to file (useful for analysis)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Save artifacts
    dump(tfidf, os.path.join(out_dir, "vectorizer.joblib"))
    dump(model, os.path.join(out_dir, "model.joblib"))
    dump(model.classes_, os.path.join(out_dir, "classes.joblib"))
    print(f"Saved model artifacts to: {out_dir}/")

if __name__ == "__main__":
    train_and_save()
