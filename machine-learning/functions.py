import time
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import re
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#import sql libraries
import sqlite3
#import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Global Parameters
stop_words = set(stopwords.words('english'))

def load_dataset(filename, cols):
    
    conn = sqlite3.connect(filename) 
    c = conn.cursor()

    query="SELECT text FROM disaster LIMIT "+str(100)
    sql_query = pd.read_sql_query (query, conn)

    dataset = pd.DataFrame(sql_query, columns = ['text'])
    
    
    #dataset = pd.read_csv(filename, encoding='latin-1')
    # dataset.to_pickle('training.pkl')   
    #dataset = pd.read_pickle('training.pkl')
    
    dataset.columns = cols
    return dataset

def load_dataset_db(filename):
    
    conn = sqlite3.connect(filename) 
    c = conn.cursor()

    query="SELECT text FROM disaster" 
    
    sql_query = pd.read_sql_query (query, conn)

    dataset = pd.DataFrame(sql_query, columns = ['text'])
    
    
    #dataset = pd.read_csv(filename, encoding='latin-1')
    # dataset.to_pickle('training.pkl')   
    #dataset = pd.read_pickle('training.pkl')
    
    # dataset.columns = cols
    return dataset


def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset

def preprocess_tweet_text(tweet):
    #tweet=str(tweet)
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove RT from tweets
    tweet = re.sub(r'RT','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # tweet=TextBlob(tweet)
    # tweet=tweet.translate(to='en')
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    #ps = PorterStemmer()
    #stemmed_words = [ps.stem(w) for w in filtered_words]
    #lemmatizer = WordNetLemmatizer()
    #lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    
    return " ".join(filtered_words)

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

def load_dataset(filename, cols):
    
    conn = sqlite3.connect(filename) 
    c = conn.cursor()

    sql_query = pd.read_sql_query ('''
                               SELECT
                               *
                               FROM disaster
                               ''', conn)

    dataset = pd.DataFrame(sql_query, columns = ['text'])
    
    
    #dataset = pd.read_csv(filename, encoding='latin-1')
    # dataset.to_pickle('training.pkl')   
    #dataset = pd.read_pickle('training.pkl')
    
    dataset.columns = cols
    return dataset

def get_sentiment(x):
    sentiment = TextBlob(x)
    return sentiment.sentiment.polarity