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

for i in tqdm(range(0, len(news_articles), batch_size)):
    batch_articles = news_articles[i:i+batch_size]
    try:
        with torch.amp.autocast('cuda'):
            # Classify articles in batch
            results = classifier(batch_articles, target_names)
            for result in results:
                predicted_label = result['labels'][0]  # Top predicted category
                predicted_categories.append(predicted_label)
    except Exception as e:
        predicted_categories.extend([None] * len(batch_articles))  # In case of failure, add None for each article in the batch


# Map predicted category labels to numeric values
predicted_category_indices = [target_names.index(cat) if cat in target_names else None for cat in predicted_categories]

# Remove articles where predictions failed (None values)
valid_indices = [i for i, pred in enumerate(predicted_category_indices) if pred is not None]
actual = np.array(actual_categories)[valid_indices]
predicted = np.array(predicted_category_indices)[valid_indices]

# Calculate accuracy
accuracy = accuracy_score(actual, predicted)
print(f"Accuracy of BART-large-mnli model: {accuracy * 100:.2f}%")
