import pandas as pd
from dask import dataframe as ddf
import time

# csv pandas read time
start_time=time.time()
#df=pd.read_csv('/Users/bharat/Documents/python_twitter/twitter-analysis/disaster-tweets.csv',encoding='latin-1',lineterminator='\n')
df=pd.read_csv('blackfriday.csv')

time_csv=(time.time()-start_time)
print(time_csv)

# csv dask read time
start_time=time.time()
#dask_df=ddf.read_csv('/Users/bharat/Documents/python_twitter/twitter-analysis/disaster-tweets.csv',encoding='latin-1',lineterminator='\n', parse_dates=["DATETIME"], blocksize=1000000,)
dask_df=ddf.read_csv('blackfriday.csv')
time_dask=(time.time()-start_time)
print(time_csv)

