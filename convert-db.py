import pandas as pd
import sqlite3
import numpy as np
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import nltk
import re
import string
# from dask import dataframe as ddf
import matplotlib.pyplot as plt
import plotly.express as px
import time
# nltk.download('stopwords')
# nltk.download('punkt')

stop_words = set(stopwords.words('english'))

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

def get_sentiment(x):
    sentiment = TextBlob(x)
    return sentiment.sentiment.polarity

while True:

    # code goes here
    time.sleep(30)

    conn = sqlite3.connect('data/dataset.db') 
    c = conn.cursor()

    query="SELECT text,id FROM disaster"
    sql_query = pd.read_sql_query (query, conn)

    dataset = pd.DataFrame(sql_query, columns = ['text','id'])

# print(dataset)    
    # print(dataset)
# # dataset = pd.read_csv(filename, encoding='latin-1')
# dataset.to_csv('dataset.csv')
    # dataset.to_parquet('data/dataset.parquet')
    # print("converted to parquet")   
# #dataset = pd.read_pickle('training.pkl')
    

# # dataset_columns = ["target", "ids", "date", "flag", "user", "text"]
# # dataset=pd.read_csv('testdataset.csv',encoding='latin-1',lineterminator="\n",names=dataset_columns)

# print(dataset)
# dataset.to_parquet('testdataset.parquet')
    dataset=dataset.dropna()
    records=len(dataset)
    print(records)
    arr=[]
    #steps=1
    # for i in range(0,records+1,int(records/steps)):
    #     arr.append(i)

    print(arr)
    number_of_rows=[]
    time_taken_lr=[]
    accuracy_lr=[]
    time_taken_nb=[]
    accuracy_nb=[]
    
    # dataset_reset=dataset
    # print(i)
    
    # n_dataset = remove_unwanted_cols(dataset, ['t_id', 'created_at', 'query', 'user'])
    print(dataset.count)
    dataset.text = dataset['text'].apply(preprocess_tweet_text)

    dataset['target'] = np.random.randint(0, 1, dataset.shape[0])
    dataset=dataset[['target','text']]

    dataset['sentiment'] = dataset['text'].apply(get_sentiment)
    dataset['target'][dataset['sentiment']>0] = 1
    dataset['target'][dataset['sentiment']<0] = -1
    dataset['target'][dataset['sentiment']==0] = 0

    print(dataset)
    #sentiment score histogram
    fig, ax = plt.subplots()
    dataset.hist('target', ax=ax)
    fig.savefig('flask-api/static/histogram_sentiment.png')
    dataset.to_parquet('data/dataset_cleaned.parquet')



    time.sleep(120)
