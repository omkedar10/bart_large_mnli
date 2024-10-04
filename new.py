import torch
from sklearn.datasets import fetch_20newsgroups
from transformers import pipeline
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # For progress tracking
from torch.cuda.amp import autocast

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print("Using GPU")
else:
    print("Using CPU")

# Load the dataset
newsgroups = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
news_articles = newsgroups.data[:100]
actual_categories = newsgroups.target[:100]
target_names = newsgroups.target_names

print("Loaded dataset with the following categories:")
print(target_names)

# Load BART model for zero-shot classification on GPU (if available)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Batch processing of news articles
batch_size = 64  # You can adjust this number based on GPU memory size
predicted_categories = []

# Ensure the batch size does not exceed the number of articles
num_batches = len(news_articles) // batch_size + (len(news_articles) % batch_size > 0)

for i in tqdm(range(num_batches)):
    batch_articles = news_articles[i * batch_size:(i + 1) * batch_size]

    if len(batch_articles) == 0:
        continue  # Skip if no articles are left to classify

    try:
        with autocast():
            # Classify articles in batch
            results = classifier(batch_articles, target_names)
            for result in results:
                predicted_label = result['labels'][0]  # Top predicted category
                predicted_categories.append(predicted_label)
    except Exception as e:
        print(f"Error during classification: {e}")
        predicted_categories.extend([None] * len(batch_articles))  # Add None for each article in case of failure

# Map predicted category labels to numeric values
predicted_category_indices = [
    target_names.index(cat) if cat in target_names else None for cat in predicted_categories
]

# Remove articles where predictions failed (None values) and out-of-bounds indices
valid_indices = [
    i for i, pred in enumerate(predicted_category_indices)
    if pred is not None and 0 <= pred < len(target_names)
]
actual = np.array(actual_categories)[valid_indices]
predicted = np.array(predicted_category_indices)[valid_indices]

# Debugging output to check contents of actual and predicted
print("Actual categories:", actual)
print("Predicted categories:", predicted)

# Calculate accuracy only if there are valid predictions
if len(predicted) > 0 and len(actual) > 0:
    min_length = min(len(predicted), len(actual))
    accuracy = np.mean(predicted[:min_length] == actual[:min_length])
    print(f"Accuracy of BART-large-mnli model: {accuracy * 100:.2f}%")
else:
    print("No valid predictions or actual categories to compute accuracy.")
