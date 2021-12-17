import pandas as pd
import time 

import sqlite3

while(True):
    
    conn = sqlite3.connect('/Users/bharat/Documents/python_twitter/twitter-analysis/flask-api/disaster-tweets-load.db')
    c = conn.cursor()
    sql_query = pd.read_sql_query ('''
                               SELECT
                               *
                               FROM disaster
                               ''', conn)

    # print("sql load time")
    start_time = time.time()
    dataset = pd.DataFrame(sql_query, columns = ['text'])
    sql_load_time=(time.time() - start_time)



#write to parquet
    dataset.to_parquet("disaster-tweets.parquet") 
    dataset.to_pickle("disaster-tweets.pkl") 

#write to csv
    dataset.to_csv('disaster-tweets.csv')

    def load_dataset_parquet(filename):
      
        dataset = pd.read_parquet(filename,engine='pyarrow')
    
        return dataset

    def load_dataset_csv(filename):
    
        dataset = pd.read_csv(filename,encoding='latin-1',lineterminator='\n')
        return dataset

    def load_dataset_pickle(filename):
    
        dataset = pd.read_pickle(filename)
        return dataset


    start_time = time.time()
    dataset= load_dataset_csv("disaster-tweets.csv")
    csv_load_time=(time.time() - start_time)


    start_time = time.time()
    dataset = load_dataset_parquet("disaster-tweets.parquet")
    parquet_load_time=(time.time() - start_time)

    start_time = time.time()
    dataset = load_dataset_pickle("disaster-tweets.pkl")
    pickle_load_time=(time.time() - start_time)

    arr=[[len(dataset),sql_load_time,parquet_load_time,pickle_load_time,csv_load_time]]
    load_times = pd.DataFrame(arr, columns = ['no_of_rows','sql', 'parquet','pickle','csv'])
    load_times.to_csv('load_times.csv', mode='a', header=False)
    
    df=pd.read_csv('load_times.csv')
    print(df)
    print((df.set_index(df.columns[0]).diff()))
    time.sleep(180)


