# gpt last version
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import pandas as pd  # Ensure pandas is imported

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# Preprocess text data
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


# Flatten the dataset if needed
def flatten_dataset(dataset):
    if isinstance(dataset, pd.DataFrame):
        # Convert DataFrame to list, assuming relevant data is in the first column
        dataset = dataset.iloc[:, 0].tolist()
    if isinstance(dataset[0], list):
        return [' '.join(entry) for entry in dataset]
    return dataset


# Function to get the average vector for a sentence
def get_average_vector(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


# Function to dynamically determine threshold based on text length
def determine_threshold(text_length):
    if text_length < 100:
        return 0.2
    elif text_length < 500:
        return 0.23
    elif text_length < 1000:
        return 0.28
    elif text_length < 1500:
        return 0.32
    elif text_length < 2000:
        return 0.36
    else:
        return 0.4


# Modified function to check for conflicts with Quranic verses and adjust overall_score
# def check_conflict_quran(input_text, quran_verses, threshold, overall_score):
#     information_sentences = sent_tokenize(input_text)
#     quran_verses_processed = [preprocess_text(verse) for verse in quran_verses]
#     information_processed = [preprocess_text(sentence) for sentence in information_sentences]
#     model = Word2Vec(quran_verses_processed, vector_size=100, window=5, min_count=1, workers=4)
#     quran_verses_embeddings = [get_average_vector(verse, model) for verse in quran_verses_processed]
#     information_embeddings = [get_average_vector(sentence, model) for sentence in information_processed]

#     for info_vec in information_embeddings:
#         similarities = [cosine_similarity([info_vec], [verse_vec])[0][0] for verse_vec in quran_verses_embeddings]
#         max_similarity = max(similarities)
#         print(f"conflict max_similarity: {max_similarity}")
#         if max_similarity > threshold:
#             if max_similarity >= 0.9:
#                 overall_score -= 40
#             elif max_similarity >= 0.86:
#                 overall_score -= 37
#             elif max_similarity >= 0.82:
#                 overall_score -= 34
#             elif max_similarity >= 0.78:
#                 overall_score -= 31
#             elif max_similarity >= 0.74:
#                 overall_score -= 28
#             elif max_similarity >= 0.70:
#                 overall_score -= 25
#             elif max_similarity >= 0.66:
#                 overall_score -= 22
#             elif max_similarity >= 0.62:
#                 overall_score -= 19
#             elif max_similarity >= 0.58:
#                 overall_score -= 16
#             elif max_similarity >= 0.54:
#                 overall_score -= 13
#             elif max_similarity >= 0.50:
#                 overall_score -= 10
#             elif max_similarity >= 0.46:
#                 overall_score -= 7
#             elif max_similarity >= 0.42:
#                 overall_score -= 4

#             elif max_similarity >= 0.38:
#                 overall_score += 7
#             elif max_similarity >= 0.34:
#                 overall_score += 9
#             elif max_similarity >= 0.30:
#                 overall_score += 11
#             elif max_similarity >= 0.26:
#                 overall_score += 13
#             elif max_similarity >= 0.22:
#                 overall_score += 15
#             elif max_similarity >= 0.18:
#                 overall_score += 17
#             elif max_similarity >= 0.15:
#                 overall_score += 19
#             else:
#                 overall_score += 21

#             break  # Exit loop once a conflict is found

#     return overall_score


def check_conflict_quran(input_text, quran_verses, threshold, overall_score):
    # Process the entire input text as a single entity
    input_processed = preprocess_text(input_text)
    quran_verses_processed = [preprocess_text(verse) for verse in quran_verses]

    # Train the Word2Vec model on Quranic verses
    model = Word2Vec(quran_verses_processed, vector_size=100, window=5, min_count=1, workers=4)

    # Get embeddings for the entire input text and Quranic verses
    input_embedding = get_average_vector(input_processed, model)
    quran_verses_embeddings = [get_average_vector(verse, model) for verse in quran_verses_processed]

    # Calculate similarities between the input text and each Quranic verse
    similarities = [cosine_similarity([input_embedding], [verse_vec])[0][0] for verse_vec in quran_verses_embeddings]

    # Find the maximum similarity
    max_similarity = max(similarities)
    # print(f"conflict max_similarity: {max_similarity}")

    # Adjust the overall_score based on the max_similarity
    if max_similarity > threshold:
        if max_similarity >= 0.9:
            overall_score -= 40
        elif max_similarity >= 0.86:
            overall_score -= 37
        elif max_similarity >= 0.82:
            overall_score -= 34
        elif max_similarity >= 0.78:
            overall_score -= 31
        elif max_similarity >= 0.74:
            overall_score -= 28
        elif max_similarity >= 0.70:
            overall_score -= 25
        elif max_similarity >= 0.66:
            overall_score -= 22
        elif max_similarity >= 0.62:
            overall_score -= 19
        elif max_similarity >= 0.58:
            overall_score -= 16
        elif max_similarity >= 0.54:
            overall_score -= 13
        elif max_similarity >= 0.50:
            overall_score -= 10
        elif max_similarity >= 0.46:
            overall_score -= 7
        elif max_similarity >= 0.42:
            overall_score -= 4

        elif max_similarity >= 0.38:
            overall_score += 7
        elif max_similarity >= 0.34:
            overall_score += 9
        elif max_similarity >= 0.30:
            overall_score += 11
        elif max_similarity >= 0.26:
            overall_score += 13
        elif max_similarity >= 0.22:
            overall_score += 15
        elif max_similarity >= 0.18:
            overall_score += 17
        elif max_similarity >= 0.15:
            overall_score += 19
        else:
            overall_score += 21

    return overall_score


# Modified function to check similarity with a given dataset and adjust overall_score
def check_similarity(input_text, dataset, threshold, overall_score):
    dataset_processed = [preprocess_text(entry) for entry in dataset]
    input_processed = preprocess_text(input_text)
    model = Word2Vec(dataset_processed, vector_size=100, window=5, min_count=1, workers=4)
    dataset_embeddings = [get_average_vector(entry, model) for entry in dataset_processed]
    input_embedding = get_average_vector(input_processed, model)

    similarities = [cosine_similarity([input_embedding], [entry_vec])[0][0] for entry_vec in dataset_embeddings]
    max_similarity = max(similarities)
    # print(f"max_similarity: {max_similarity}")
    if max_similarity > threshold:
        if max_similarity >= 0.9:
            overall_score += 30

        elif max_similarity >= 0.86:
            overall_score += 28
        elif max_similarity >= 0.80:
            overall_score += 26
        elif max_similarity >= 0.74:
            overall_score += 23
        elif max_similarity >= 0.68:
            overall_score += 20
        elif max_similarity >= 0.65:
            overall_score += 18
        elif max_similarity >= 0.62:
            overall_score += 16
        elif max_similarity >= 0.57:
            overall_score += 14
        elif max_similarity >= 0.52:
            overall_score += 12
        elif max_similarity >= 0.48:
            overall_score += 9
        elif max_similarity >= 0.40:
            overall_score += 7
        elif max_similarity >= 0.35:
            overall_score += 4
        elif max_similarity >= 0.30:
            overall_score += 1

        elif max_similarity >= 0.28:
            overall_score -= 7
        elif max_similarity >= 0.25:
            overall_score -= 9
        elif max_similarity >= 0.21:
            overall_score -= 11
        elif max_similarity >= 0.18:
            overall_score -= 13
        elif max_similarity >= 0.15:
            overall_score -= 15
        elif max_similarity >= 0.12:
            overall_score -= 17
        elif max_similarity >= 0.1:
            overall_score -= 19
        else:
            overall_score -= 21

    return overall_score


def check_fake_news_similarity(input_text, fake_news_dataset, threshold, overall_score):
    fake_news_processed = [preprocess_text(entry) for entry in fake_news_dataset]
    input_processed = preprocess_text(input_text)
    model = Word2Vec(fake_news_processed, vector_size=100, window=5, min_count=1, workers=4)
    fake_news_embeddings = [get_average_vector(entry, model) for entry in fake_news_processed]
    input_embedding = get_average_vector(input_processed, model)

    similarities = [cosine_similarity([input_embedding], [entry_vec])[0][0] for entry_vec in fake_news_embeddings]
    max_similarity = max(similarities)
    # print(f"Fake News max_similarity: {max_similarity}")
    if max_similarity > threshold:
        if max_similarity >= 0.9:
            overall_score -= 30

        elif max_similarity >= 0.86:
            overall_score -= 28
        elif max_similarity >= 0.80:
            overall_score -= 26
        elif max_similarity >= 0.74:
            overall_score -= 23
        elif max_similarity >= 0.68:
            overall_score -= 20
        elif max_similarity >= 0.65:
            overall_score -= 18
        elif max_similarity >= 0.62:
            overall_score -= 16
        elif max_similarity >= 0.57:
            overall_score -= 14
        elif max_similarity >= 0.52:
            overall_score -= 12
        elif max_similarity >= 0.48:
            overall_score -= 9
        elif max_similarity >= 0.40:
            overall_score -= 7
        elif max_similarity >= 0.35:
            overall_score -= 4
        elif max_similarity >= 0.30:
            overall_score -= 1

        elif max_similarity >= 0.28:
            overall_score += 7
        elif max_similarity >= 0.25:
            overall_score += 9
        elif max_similarity >= 0.21:
            overall_score += 11
        elif max_similarity >= 0.18:
            overall_score += 13
        elif max_similarity >= 0.15:
            overall_score += 15
        elif max_similarity >= 0.12:
            overall_score += 17
        elif max_similarity >= 0.1:
            overall_score += 19
        else:
            overall_score += 21

    return overall_score


# Main function
def main_func(input_text, source_information, fake_news, quran_verses):
    overall_score = 50  # Starting score

    # Determine threshold based on text length
    text_length = len(input_text)
    threshold = determine_threshold(text_length)

    # Flatten datasets if necessary
    source_information = flatten_dataset(source_information)
    fake_news = flatten_dataset(fake_news)
    quran_verses = flatten_dataset(quran_verses)

    # Check conflict with Quran and adjust overall_score
    overall_score = check_conflict_quran(input_text, quran_verses, threshold, overall_score)
    # print(f"Overall score after check_conflict_quran: {overall_score}")

    # Check similarity with source information and adjust overall_score
    overall_score = check_similarity(input_text, source_information, threshold, overall_score)
    # print(f"Overall score after check_similarity (source_information): {overall_score}")

    # Check similarity with fake news and adjust overall_score
    overall_score = check_fake_news_similarity(input_text, fake_news, threshold, overall_score)
    # print(f"Overall score after check_similarity (fake_news): {overall_score}")

    # Ensure overall_score is within 0 to 100
    overall_score = max(0, min(overall_score, 100))

    return overall_score


# # Load datasets
with open('source_information_processed.pkl', 'rb') as f:
    source_information = pickle.load(f)

with open('fake_news_processed.pkl', 'rb') as f:
    fake_news = pickle.load(f)

with open('Quran_verses.pkl', 'rb') as f:
    quran_verses = pickle.load(f)
