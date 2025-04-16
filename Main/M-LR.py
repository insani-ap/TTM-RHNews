import itertools
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

raw_file = os.path.join("Dataset", "Mafindo", "Mafindo.csv")
cleaned_file = os.path.join("Dataset", "Mafindo", "MafindoCleaned.csv")

raw_data = pd.read_csv(raw_file)
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

#Param Combination
penalties = ['l1', 'l2', 'elasticnet']
C_values = [0.1, 1, 10, 100]
solvers = ['liblinear', 'saga', 'sag', 'newton-cg', 'lbfgs']
fit_intercepts = [True, False]
warm_starts = [True, False]
tols = [1e-4, 1e-3, 1e-2, 1e-1]
max_iter = [100, 250, 500]

#Combination
param_combinations = list(itertools.product(penalties, C_values, solvers,
                                            fit_intercepts, warm_starts, tols, max_iter))

#Report
for penalty, C, solver, fit_intercept, warm_start, tol, max_iter in param_combinations:
  try:
    model = LogisticRegression(penalty=penalty, C=C, solver=solver,
                               fit_intercept=fit_intercept,
                               warm_start=warm_start, tol=tol, max_iter=max_iter)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Append results along with confusion matrix and other metrics
    results.append((penalty, C, solver, fit_intercept, warm_start, tol, max_iter, accuracy, precision, recall, f1, cm))

  except Exception as e:
    print(f"Error with params: penalty={penalty}, C={C}, solver={solver}, "
          f"fit_intercept={fit_intercept}, warm_start={warm_start}, "
          f"tol={tol}: {e}, max_iter={max_iter}")

# Sort results by accuracy (descending)
sorted_results = sorted(results, key=lambda x: x[7], reverse=True)

# Print sorted results
print("\nSorted Results (by Accuracy):")
for result in sorted_results:
    print(f"Params: penalty={result[0]}, C={result[1]}, solver={result[2]}, "
          f"fit_intercept={result[3]}, warm_start={result[4]}, "
          f"tol={result[5]}, max_iter={result[6]}")
    print(f"Accuracy: {result[7]}, Precision: {result[8]}, Recall: {result[9]}, F1 Score: {result[10]}")
    print(f"Confusion Matrix:\n{result[11]}")
    print("-" * 50)

# Best Hyperparameters
best_params = sorted_results[0][:7] if sorted_results else None
best_accuracy = sorted_results[0][7] if sorted_results else 0

print("\nBest Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)