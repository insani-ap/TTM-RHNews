import os
import pandas as pd
from collections import Counter
import string

# File paths
raw_file = os.path.join("Dataset", "TTB", "TTB.csv")
cleaned_file = os.path.join("Dataset", "TTB", "TTBCleaned.csv")

# Load datasets
raw_data = pd.read_csv(raw_file, encoding='unicode_escape')
cleaned_data = pd.read_csv(cleaned_file)

# Rename first column to 'data' for clarity
cleaned_data.rename(columns={cleaned_data.columns[0]: 'data'}, inplace=True)

# Add 'isHoax' column from raw_data to cleaned_data
cleaned_data['isHoax'] = raw_data['isHoax']

# Filter only hoax data
hoax_data = cleaned_data[cleaned_data['isHoax'] == 1]['data'].dropna()

# Initialize Counter for word counts
word_count = Counter()

# Process each line in hoax_data
for line in hoax_data:
    line = line.strip().lower()  # Remove leading/trailing spaces and lowercase
    # Remove punctuation and split into words
    words = line.translate(str.maketrans('', '', string.punctuation)).split()
    word_count.update(words)

# Sort word counts in descending order
sorted_word_count = word_count.most_common()

# Print top 20 words
print("Top 20 words:")
for word, count in sorted_word_count[:20]:
    print(f"{word}: {count}")

# Optional: Save word counts to a CSV file
output_file = os.path.join("Dataset", "TTB", "hoax_word_counts.csv")
pd.DataFrame(sorted_word_count, columns=["Word", "Count"]).to_csv(output_file, index=False)
print(f"Word counts saved to {output_file}")