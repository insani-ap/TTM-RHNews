import itertools
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

raw_file = os.path.join("Dataset", "TTB", "TTB.csv")
cleaned_file = os.path.join("Dataset", "TTB", "TTBCleaned.csv")

raw_data = pd.read_csv(raw_file, encoding='unicode_escape')
cleaned_data = pd.read_csv(cleaned_file)
cleaned_data.rename(columns={cleaned_data.columns[0]: 'data'}, inplace=True)

labels = raw_data['isHoax']

vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(cleaned_data['data'])

x_train, x_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    train_size=0.8)

# List to store all results
results = []

# Param Combination
alpha = [0.001, 0.01, 0.1, 0.5, 1]
fit_prior = [True, False]
force_alpha = [True, False]

# Combination
param_combinations = list(itertools.product(alpha, fit_prior, force_alpha))

# Report
for alpha, fit_prior, force_alpha in param_combinations:
    try:
        model = MultinomialNB(alpha=alpha, fit_prior=fit_prior, force_alpha=force_alpha)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Append results along with confusion matrix and other metrics
        results.append((alpha, fit_prior, force_alpha, accuracy, precision, recall, f1, cm))

    except Exception as e:
        print(f"Error with params: alpha={alpha}, fit_prior={fit_prior}, force_alpha={force_alpha}: {e}")

# Sort results by accuracy (descending)
sorted_results = sorted(results, key=lambda x: x[3], reverse=True)

# Print sorted results
print("\nSorted Results (by Accuracy):")
for result in sorted_results:
    print(f"Params: alpha={result[0]}, fit_prior={result[1]}, force_alpha={result[2]}, Accuracy: {result[3]}")
    print(f"Precision: {result[4]}, Recall: {result[5]}, F1 Score: {result[6]}")
    print(f"Confusion Matrix:\n{result[7]}")
    print("-" * 50)

# Best Hyperparameters
best_params = sorted_results[0][:3] if sorted_results else None
best_accuracy = sorted_results[0][3] if sorted_results else 0

print("\nBest Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)