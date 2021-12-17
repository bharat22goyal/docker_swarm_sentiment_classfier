# print('new')
import time
# start_time_overall=time.time()
# import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


print('start')
def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector



while(True):
    print('start')
    path='data/dataset_cleaned.parquet'
    print('read')
    if os.path.isfile(path)!=True:
         time.sleep(600)
    dataset=pd.read_parquet(path)


    time_taken_lr=[]
    accuracy_lr=[]
    time_taken_nb=[]
    accuracy_nb=[]

    print('middle')
    # Same tf vector will be used for Testing sentiments on unseen trending data
    tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
    X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
    y = np.array(dataset.iloc[:, 0]).ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

        # Training Naive Bayes model
    start_time_nb = time.time()
    NB_model = MultinomialNB()
    NB_model.fit(X_train, y_train)
    y_predict_nb = NB_model.predict(X_test)
    time_taken_nb=(time.time() - start_time_nb)
    accuracy_nb=accuracy_score(y_test, y_predict_nb)
    # print(accuracy_nb)


        # # Training Logistics Regression model
    start_time_lr = time.time()
    LR_model = LogisticRegression(solver='lbfgs',max_iter = 1000)
    LR_model.fit(X_train, y_train)
    y_predict_lr = LR_model.predict(X_test)
    time_taken_lr=(time.time() - start_time_lr)
    accuracy_lr=accuracy_score(y_test, y_predict_lr)
        


    eval_acc=[[len(dataset),accuracy_lr,accuracy_nb]]
    eval_time=[[len(dataset),time_taken_lr,time_taken_nb]]
        
    eval_results_acc = pd.DataFrame(eval_acc, columns = ['records','lr','nb'])
    eval_results_time = pd.DataFrame(eval_time, columns = ['records','time_lr','time_nb'])

    eval_results_acc.to_csv('eval_results_acc.csv', mode='a', header=False)
    eval_results_time.to_csv('eval_results_time.csv', mode='a', header=False)

    df_eval=pd.read_csv('eval_results_acc.csv',
                    names=['a','records','lr','nb'])
    # del df_eval['a']
    print(df_eval)
    df_time=pd.read_csv('eval_results_time.csv',
                        names=['a','records','lr','nb'])
    # del df_eval['a']

    fig,ax = plt.subplots()
    plt.ticklabel_format(style='plain')
    ax.plot(df_eval['records'], df_eval['lr'], color='red',label='lr',marker='o')
    plt.title('Accuracy comparison', fontsize=14)
    ax.set_xlabel('records', fontsize=14)
    ax.set_ylabel('accuracy(%)', fontsize=14)
    ax2=ax.twinx()
    ax2.plot(df_eval['records'], df_eval['nb'], color='blue', marker='o',label='nb')
    ax2.set_ylabel("accuracy(%)",fontsize=14)
    ax.legend(loc="lower right")
    ax2.legend(loc="upper left")
    plt.grid(True)
    plt.show()
    fig.savefig('flask-api/static/eval.jpeg')

    fig,ax = plt.subplots()
    plt.ticklabel_format(style='plain')
    ax.plot(df_eval['records'], df_time['lr'], color='red',label='lr',marker='o')
    plt.title('Time comparison', fontsize=14)
    ax.set_xlabel('records', fontsize=14)
    ax.set_ylabel('time(s)', fontsize=14)
    ax2=ax.twinx()
    ax2.plot(df_eval['records'], df_time['nb'], color='blue', marker='o',label='nb')
    ax2.set_ylabel("time(s)",fontsize=14)
    ax.legend(loc="lower right")
    ax2.legend(loc="upper left")
    plt.grid(True)
    plt.show()
    fig.savefig('flask-api/static/time.jpeg')

    time.sleep(5)


