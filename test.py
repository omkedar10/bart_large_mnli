import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline
from datasets import Dataset

# Check if GPU is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Example dataset (replace with your actual dataset)
news_articles = [
    "The stock market crashed due to economic instability.",
    "Scientists discovered a new species of dinosaur.",
    "The local team won the championship against their rivals.",
    "A recent study shows that climate change is accelerating.",
    "The new smartphone model has innovative features.",
    "This year's elections will decide the future of the country.",
    # Add more articles as needed
]

# Define categories for classification
target_names = [
    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
    'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
    'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
    'sci.space', 'soc.religion.christian', 'talk.politics.guns',
    'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
]

# Create a Dataset object from the list of articles
dataset = Dataset.from_dict({"text": news_articles})

# Batch processing of news articles
batch_size = 5  # Adjust this number based on GPU memory size and dataset size
predicted_categories = []

# Process the dataset in batches
for i in tqdm(range(0, len(dataset), batch_size)):
    batch_articles = dataset[i:i + batch_size]['text']  # Accessing batch articles
    try:
        # Classify articles in batch
        results = classifier(batch_articles, candidate_labels=target_names)

        # Process results
        for result in results:
            predicted_label = result['labels'][0]  # Top predicted category
            predicted_categories.append(predicted_label)

    except Exception as e:
        print(f"Error processing batch {i // batch_size}: {e}")
        predicted_categories.extend([None] * len(batch_articles))  # Add None for failed predictions

# Map predicted categories to numeric values and compute accuracy
predicted_categories = np.array(predicted_categories)
# Replace actual_categories with your actual categories
actual_categories = np.array(['talk.politics.misc', 'sci.med', 'rec.sport.baseball', 'sci.space', 'comp.graphics', 'talk.politics.guns'])

# Print predictions and actual categories
print("Predicted Categories:", predicted_categories)
print("Actual Categories:", actual_categories)

# Calculate accuracy if both arrays have the same length
if predicted_categories.size > 0 and actual_categories.size > 0:
    min_length = min(len(predicted_categories), len(actual_categories))
    accuracy = np.mean(predicted_categories[:min_length] == actual_categories[:min_length])
    print(f"Accuracy of BART-large-mnli model: {accuracy * 100:.2f}%")
else:
    print("No valid predictions or actual categories to compute accuracy.")

