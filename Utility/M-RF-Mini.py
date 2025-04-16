import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

raw_file = os.path.join("Dataset", "Mafindo", "Mafindo.csv")
cleaned_file = os.path.join("Dataset", "Mafindo", "MafindoCleaned.csv")

raw_data = pd.read_csv(raw_file)
cleaned_data = pd.read_csv(cleaned_file)
cleaned_data.rename(columns={cleaned_data.columns[0]: "data"}, inplace=True)

labels = raw_data["isHoax"]

vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(cleaned_data["data"])

x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, train_size=0.8
)

# Initialize Variables
results = []

# Define the specific hyperparameters to test
param_combinations = [
    (100, None, 10, 1, "sqrt", False, False, "gini", 0.0, -1, 50),
    (100, None, 5, 2, "log2", True, True, "gini", 0.0, 1, 50),
    (100, 20, 10, 4, "sqrt", True, False, "entropy", 0.01, -1, 100),
    (200, 10, 2, 1, "log2", False, False, "gini", 0.1, 1, 50),
    (200, 10, 5, 2, "log2", False, False, "gini", 0.01, -1, 50),
    (100, 10, 2, 4, "log2", True, True, "gini", 0.1, -1, 100),
    (200, 30, 2, 2, "log2", False, False, "gini", 0.01, 1, 100),
    (100, 30, 5, 4, "log2", True, False, "entropy", 0.1, -1, 50),
    (200, 30, 10, 4, "log2", False, False, "entropy", 0.1, 1, 50),
    (300, 30, 10, 4, "log2", False, False, "entropy", 0.1, 1, 100),
]

# Model Training and Evaluation
for n_est, max_d, min_split, min_leaf, max_feat, boot, oob, crit, min_impurity, jobs, rand_state in param_combinations:
    try:
        # Initialize Model
        model = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            max_features=max_feat,
            bootstrap=boot,
            oob_score=oob,
            criterion=crit,
            min_impurity_decrease=min_impurity,
            random_state=rand_state,
            n_jobs=jobs,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Store Results
        results.append(
            (n_est, max_d, min_split, min_leaf, max_feat, boot, oob, crit, min_impurity, jobs, rand_state, accuracy, precision, recall, f1, cm)
        )

    except Exception as e:
        print(f"Error with params: n_estimators={n_est}, max_depth={max_d}, min_samples_split={min_split}, "
              f"min_samples_leaf={min_leaf}, max_features={max_feat}, bootstrap={boot}, oob_score={oob}, "
              f"criterion={crit}, min_impurity_decrease={min_impurity}, n_jobs={jobs}, random_state={rand_state}: {e}")

# Sort results by accuracy (descending)
sorted_results = sorted(results, key=lambda x: x[11], reverse=True)

# Print sorted results
print("\nSorted Results (by Accuracy):")
for result in sorted_results:
    print(f"Params: n_estimators={result[0]}, max_depth={result[1]}, min_samples_split={result[2]}, "
          f"min_samples_leaf={result[3]}, max_features={result[4]}, bootstrap={result[5]}, oob_score={result[6]}, "
          f"criterion={result[7]}, min_impurity_decrease={result[8]}, n_jobs={result[9]}, random_state={result[10]}")
    print(f"Accuracy: {result[11]}, Precision: {result[12]}, Recall: {result[13]}, F1 Score: {result[14]}")
    print(f"Confusion Matrix:\n{result[15]}")
    print("-" * 50)

# Best Hyperparameters
best_params = sorted_results[0][:11] if sorted_results else None
best_accuracy = sorted_results[0][11] if sorted_results else 0

print("\nBest Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)