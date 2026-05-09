import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from collections import defaultdict
from joblib import Parallel, delayed
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import numpy as np
import pickle
import re
import os

PREPROCESSED_CACHE = "preprocessed_cache.pkl"
TFIDF_FEATURES = 3000 # Number of important words found in data to use as features for ML models


stop_words = set(stopwords.words('english'))

def split_into_map(data):
     rating_map = defaultdict(list)

     for item in data:
           rating_map[item['rating']].append(item['text'])
           return rating_map


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)      
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    words = text.lower().split()                         
    words = [w for w in words if w not in stop_words]   
    return ' '.join(words)

def analyze_sentiment(data):
    print("Analyzing sentiment with VADER... ")
    analyzer = SentimentIntensityAnalyzer()
    
    def analyze_single(item):
        
        scores = analyzer.polarity_scores(item['text'])
        return [scores['compound'], scores['pos'], scores['neg'], scores['neu']]
    
    features = Parallel(n_jobs=-1)(delayed(analyze_single)(item) for item in data)
    return features


def tfidf_vectorize(cleaned_data):
    texts = [item['text'] for item in cleaned_data]
    vectorizer = TfidfVectorizer(max_features=TFIDF_FEATURES, ngram_range=(1,2))  
    X = vectorizer.fit_transform(texts)
    return X, vectorizer



def preprocess_ratings(data, use_vader=True):
 
    if os.path.exists(PREPROCESSED_CACHE):
        print("Loading preprocessed data from cache...")
        with open(PREPROCESSED_CACHE, 'rb') as f:
            return pickle.load(f)

    print("Removing stop words, punctuation, and HTML tags...")
    cleaned_data = [{'rating': item['rating'], 'text': clean_text(item['text'])} for item in data]
    
    for review in cleaned_data[:3]:
        print(review)
        print()

    rating_map = split_into_map(cleaned_data)
    tfidf_matrix, vectorizer = tfidf_vectorize(cleaned_data)
    
    if use_vader:
        features = analyze_sentiment(cleaned_data)
        X = hstack([csr_matrix(features), tfidf_matrix])
    else:
        X = tfidf_matrix

    y = np.array([item['rating'] for item in cleaned_data])  

    with open(PREPROCESSED_CACHE, 'wb') as f:
        pickle.dump((X, y), f)
    print("Saved preprocessed cache")
    
    return X, y