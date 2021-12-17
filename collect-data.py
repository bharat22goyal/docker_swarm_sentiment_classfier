import sqlite3
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import os.path

ckey=""
csecret=""
atoken=''
asecret=''

path='data/dataset.db'
if os.path.isfile(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
else: 
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute('''CREATE TABLE disaster (text,id)''')


# Create table
# c.execute('''CREATE TABLE disaster (text,id)''')

class listener(StreamListener):

    def on_data(self, data):
        try:
            all_data = json.loads(data)
            # print(all_data.keys())
            
            tweet = all_data["text"]
            print(tweet)
            id=all_data["id"]
            #print(time)
            # location=all_data["geo"]
            #username = all_data["user"]["screen_name"]
            #location = all_data["user"]["location"]

            #write data to sql    
            c.execute("INSERT INTO disaster (text,id) VALUES (?,?)",(tweet,id))

            conn.commit()

         

            return True
        except:
            print("An exception occurred")
        
       

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
#keywords=["storm","landslide","tornado","tsunami","wildfire","earthquake","famine","cyclone","typhoon","avalanche","heatwave","hurricane","flood","music","car","diamond","juice","chips","potato","burger","pizza","book","donald","modi","football","cricket","tesla","elon","google","pfizer"]
keywords_disaster=["storm","landslide","tornado","tsunami","wildfire","earthquake","famine","cyclone","typhoon","avalanche","heatwave","hurricane","omicron","covid","corona",'vaccine','pfizer','covidshield','covaxin','who','latifi','toto']
twitterStream.filter(track=keywords_disaster)
