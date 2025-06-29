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
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([int(row[f"{lang.lower()}_hits"]) 
                      for lang in ("Spanish","Portuguese","Italian")])
            y.append(row["label"])
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="CSV with regex hits + label column")
    args = parser.parse_args()

    X, y = load_data(args.csv_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()
