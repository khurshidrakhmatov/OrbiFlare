import os
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import pandas as pd

# Function to ensure NLTK data is loaded from the local nltk_data directory
# def ensure_nltk_data():
#     # Get the directory where the current script is located
#     script_dir = os.path.dirname(os.path.abspath(__file__))
    
#     # Define the path to the nltk_data directory within the repo
#     nltk_data_path = os.path.join(script_dir, 'nltk_data')
    
#     # Verify that the nltk_data directory exists
#     if not os.path.exists(nltk_data_path):
#         raise FileNotFoundError(f"nltk_data directory not found at {nltk_data_path}")
    
#     # Optionally, verify that the punkt tokenizer is present
#     punkt_dir = os.path.join(nltk_data_path, "tokenizers", "punkt")
#     if not os.path.exists(punkt_dir):
#         raise FileNotFoundError(f"punkt tokenizer not found in {punkt_dir}")
    
#     # Add the nltk_data_path to NLTK's data search paths
#     nltk.data.path.append(nltk_data_path)

# # Call the function to set up NLTK data
# ensure_nltk_data()

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess text
def preprocess_text(text, stopwordlist):
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum() and token not in stopwordlist]

# Flatten dataset if necessary
def flatten_dataset(dataset):
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.iloc[:, 0].tolist()
    if isinstance(dataset[0], list):
        return [' '.join(entry) for entry in dataset]
    return dataset

# Get average vector for a sentence
def get_average_vector(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Determine threshold based on text length
def determine_threshold(text_length):
    thresholds = [(500, 0.2), (1500, 0.23), (2500, 0.25), (3500, 0.29), (5000, 0.32)]
    for length, threshold in thresholds:
        if text_length < length:
            return threshold
    return 0.35

# General function to adjust overall score based on similarity
def adjust_score(max_similarity, score_adjustment, overall_score, condition):
    adjustment = score_adjustment * max_similarity
    return overall_score + adjustment if condition else overall_score - adjustment

# Common function for similarity check and scoring
def check_similarity_general(input_text, dataset, threshold, overall_score, condition, score_adjustment, stopwordlist):
    dataset_processed = [preprocess_text(entry, stopwordlist) for entry in dataset]
    input_processed = preprocess_text(input_text, stopwordlist)
    model = Word2Vec(dataset_processed, vector_size=100, window=5, min_count=1, workers=4)
    input_embedding = get_average_vector(input_processed, model)
    similarities = [cosine_similarity([input_embedding], [get_average_vector(entry, model)])[0][0] for entry in dataset_processed]
    max_similarity = max(similarities)
    if max_similarity > threshold:
        overall_score = adjust_score(max_similarity, score_adjustment, overall_score, condition(max_similarity))
    return overall_score

# Main function
def main_func(input_text, source_information, fake_news, quran_verses):
    overall_score = 50
    stopwordlist = [
        'a', 'about', 'above', 'after', 'again', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be',
        'because', 'been', 'before', 'being', 'below', 'between', 'by', 'can', 'did', 'do', 'does', 'doing',
        'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her',
        'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its',
        'itself', 'just', 'll', 'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once', 'only',
        'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're', 's', 'same', 'she', 'should', 'so',
        'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
        'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
        'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you',
        'your', 'yours', 'yourself', 'yourselves'
    ]
    text_length = len(input_text)
    threshold = determine_threshold(text_length)

    # Load and flatten datasets
    source_information = flatten_dataset(source_information)
    fake_news = flatten_dataset(fake_news)
    quran_verses = flatten_dataset(quran_verses)

    # Check Quran conflict
    overall_score = check_similarity_general(
        input_text, quran_verses, threshold, overall_score,
        lambda x: x >= 0.38, 41, stopwordlist
    )

    # Check similarity with source information
    overall_score = check_similarity_general(
        input_text, source_information, threshold, overall_score,
        lambda x: x >= 0.28, 31, stopwordlist
    )

    # Check similarity with fake news
    overall_score = check_similarity_general(
        input_text, fake_news, threshold, overall_score,
        lambda x: x >= 0.28, -25, stopwordlist
    )

    # Ensure score within range
    return max(0, min(overall_score, 100))