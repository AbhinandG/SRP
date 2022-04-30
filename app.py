from datetime import datetime
import streamlit as st
import sys
from pytube import YouTube
import os
import time
import requests
from zipfile import ZipFile
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import math
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin 
import pymongo
from pymongo import MongoClient
import pprint
import json


client = MongoClient('mongodb+srv://abhi:pass@cluster0.e3vf3.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')


print("======CONNECTED TO MONGODB DATABASE SUCCESSFULLY=======")

db = client["myFirstDatabase"]
collection = db["myCollection"]


st.markdown('# **Socially Relevant Project - Automatic Video Transcript Generator**')
st.markdown('**Abhinand G (2019115003) - 6th Semester, BTech IT**')

bar = st.progress(0)


def get_yt(URL):
    video = YouTube(URL)
    yt = video.streams.get_audio_only()
    yt.download()

    bar.progress(10)


def sentiment_generator(transcri):
    saved_model=joblib.load('ML_model.pkl')
    trainin_df=pd.read_csv('training.1600000.processed.noemoticon.csv')
    print("==DATASET IMPORTED==")
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

    print("TFIDF HAS BEEN FIT")

    new_dataframe={'content':transcri, 'target':5}
    print("DATAFRAME CREATED")
    new_dataframe=pd.DataFrame(new_dataframe,index=[0])
    X_test1=new_dataframe['content']
    X_test1=vectoriser.transform(X_test1)
    predicted_score=saved_model.predict(X_test1)

    polarity=TextBlob(transcri).sentiment.polarity
    subjectivity=TextBlob(transcri).sentiment.subjectivity
    if predicted_score[0]==4:
        if(polarity==0):
            st.warning('The video has a neutral sentiment with polarity value '+str(polarity))
        else:
            st.success('The video has a positive sentiment with polarity value '+str(polarity))
        barpolarity=st.progress(0)
        barpolarity.progress(math.floor(polarity*100))
        st.text("")
        st.text("")
    else:
        st.error('The video has a negative sentiment with polarity value '+str(polarity))
        barpolarity=st.progress(0)
        barpolarity.progress(math.floor(polarity*100))
        st.text("")
        st.text("")
    if subjectivity>0.5:
        st.success('The subjectivity score is '+str(subjectivity))
        barsubjectivity=st.progress(0)
        barsubjectivity.progress(math.floor(subjectivity*100))
    else:
        st.error('The subjectivity score is '+str(subjectivity))
        barsubjectivity=st.progress(0)
        barsubjectivity.progress(math.floor(subjectivity*100))
                                                            
    st.text("")
    st.text("")
    details={"Polarity":polarity, "Subjectivity":subjectivity}

    chart_data=pd.DataFrame(details,index=[0]) 
    
    st.write("Here\'s a graph to visualise the polarity and subjectivity better - ")

    st.bar_chart(chart_data)

    mydate=datetime.datetime.now()

    record={"file_name":fileToUpload.name, 
            "transcript":transcri,
            "polarity":polarity,
            "subjectivity":subjectivity,
            "timestamp": mydate}

    rec_ins=collection.insert_one(record)
    print("Data inserted for ",rec_ins)

def transcribe_yt():
    current_dir = os.getcwd()
    for file in os.listdir(current_dir):
       #if file.endswith(".mp4"):
       if file.startswith(fileToUpload.name):
            mp4_file = os.path.join(current_dir, file)
            #print(mp4_file)
    filename = mp4_file
    bar.progress(20)
    def read_file(filename, chunk_size=5242880):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data
    headers = {'authorization': st.secrets['top_secrets']['api_key'] }
    response = requests.post('https://api.assemblyai.com/v2/upload',
                            headers=headers,
                            data=read_file(filename))
    audio_url = response.json()['upload_url']
    #st.info('3. YouTube audio file has been uploaded to AssemblyAI')
    bar.progress(30)


    endpoint = "https://api.assemblyai.com/v2/transcript"

    json = {
    "audio_url": audio_url
    }

    headers = {
        "authorization": st.secrets['top_secrets']['api_key'],
        "content-type": "application/json"
    }

    transcript_input_response = requests.post(endpoint, json=json, headers=headers)

    bar.progress(40)

    transcript_id = transcript_input_response.json()["id"]

    bar.progress(50)

    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    headers = {
        "authorization": st.secrets['top_secrets']['api_key'],
    }
    transcript_output_response = requests.get(endpoint, headers=headers)

    bar.progress(60)

    from time import sleep

    st.warning('Transcription is processing ...')

    while transcript_output_response.json()['status'] != 'completed':
        sleep(5)
        transcript_output_response = requests.get(endpoint, headers=headers)
    
    bar.progress(100)

    st.header('Generated transcripts - ')

    st.info(transcript_output_response.json()["text"])

    st.text("")
    st.text("")

    print("TRANSCRIPT GENERATED... GOING TO SENTIMENTS")
    st.warning('Generating insights. This might take a minute or two...')
    sentiment_generator(transcript_output_response.json()["text"])

   

with st.sidebar:
    st.write('**Which video do you want to analyse?**')

with st.sidebar.form(key='my_form'):
	#URL = st.text_input('Enter URL of YouTube video:')
	#submit_button = st.form_submit_button(label='Go')
    fileToUpload = st.file_uploader("Upload a custom video..", type=["mp4"])
    submit_button=st.form_submit_button(label='Go')

with st.sidebar:
    st.write('This is web application that inputs .mp4 files to generate transcripts via the AssemblyAI API, and makes use of pre-trained models to predict sentiments of the transcripts and analyses the amount of profanity used in the video.')

with st.sidebar:
    st.write('Developed by: abhinandganesh2001@gmail.com')

if submit_button:
    #get_yt(URL)
    if fileToUpload is not None:
        file_details = {"FileName":fileToUpload.name,"FileType":fileToUpload.type}
        with open(os.path.join("",fileToUpload.name),"wb") as f: 
            f.write(fileToUpload.getbuffer())         
        st.success("File has been uploaded!")

        if db.collections.count_documents({"file_name": fileToUpload.name}) != 0:
            print("ALREADY EXISTS!!")


    transcribe_yt()



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


