"""
Train a Naive Bayes model on regex feature counts.

"""

import csv
import argparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(path):
    X, y = [], []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Pull from the *_hits columns
            X.append([
                int(row["spanish_hits"]),
                int(row["portuguese_hits"]),
                int(row["italian_hits"])
            ])
            y.append(row["label"])
    return X, y

def main():
    parser = argparse.ArgumentParser(
        description="Train & evaluate a MultinomialNB on regex-hit features"
    )
    parser.add_argument("csv_file", help="Path to data/features.csv")
    args = parser.parse_args()

    X, y = load_data(args.csv_file)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, zero_division=0))

if __name__ == "__main__":
    main()
