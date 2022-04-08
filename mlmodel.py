from textblob import TextBlob
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline 
from sklearn.base import TransformerMixin 
import pickle
import joblib


print("==PACKAGES IMPORTED==")

trainin_df=pd.read_csv('training.1600000.processed.noemoticon.csv')

print("==DATASET IMPORTED==")

print(trainin_df.columns)

trains_df=pd.DataFrame()

trains_df['content']=trainin_df.iloc[:,5]

trains_df['target']=trainin_df.iloc[:,0]

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours','must', 'yourself', 'yourselves']

STOPWORDS = set(stopwordlist)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
trains_df['content'] = trains_df['content'].apply(lambda text: cleaning_stopwords(text))


def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
trains_df['content'] = trains_df['content'].apply(lambda x: cleaning_URLs(x))


def cleaning_ATS(data):
    return re.sub('RT @[\w]+',' ',data)
trains_df['content'] = trains_df['content'].apply(lambda x: cleaning_ATS(x))


def cleaning_SPL(data):
    return re.sub('[^A-Za-z]',' ',data)
trains_df['content'] = trains_df['content'].apply(lambda x: cleaning_SPL(x))


print("==DATA CLEANED==")

X=trains_df['content']
y=trains_df['target']

print("==SPLIT INTO X AND Y==")

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state =26105111)

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names()))


X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

def model_Evaluate(model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)

LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train, y_train)
model_Evaluate(LR_model)
y_predict_lr = LR_model.predict(X_test)
print("ACCURACY SCORE IS: "+str(accuracy_score(y_test, y_predict_lr)))

joblib.dump(LR_model,'ML_model.pkl')

print("==MODEL PICKLED==")
