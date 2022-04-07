import streamlit as st
from pytube import YouTube
import os
import sys
import time
import requests
from zipfile import ZipFile
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



st.markdown('# **Socially Relevant Project - Automatic Video Transcript Generator**')

st.markdown('**Abhinand G (2019115003)**')
bar = st.progress(0)


def get_yt(URL):
    video = YouTube(URL)
    yt = video.streams.get_audio_only()
    yt.download()

    bar.progress(10)


def sentiment_generator(transcri):
    polarity=TextBlob(transcri).sentiment.polarity
    subjectivity=TextBlob(transcri).sentiment.polarity
    if polarity>0:
        st.success('The video has a positive sentiment with polarity value '+str(polarity))
    else:
        st.error('The video has a negative sentiment with polarity value '+str(polarity))

    chart_data=pd.DataFrame()
    chart_data['Polarity']=polarity
    chart_data['Subjectivity']=subjectivity
    
    st.line_chart(chart_data)

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

    # 4. Transcribe uploaded audio file
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


    st.header('Output')


    st.info(transcript_output_response.json()["text"])
    sentiment_generator(transcript_output_response.json()["text"])

   

  
    yt_txt = open('yt.txt', 'w')
    yt_txt.write(transcript_output_response.json()["text"])
    yt_txt.close()


    srt_endpoint = endpoint + "/srt"
    srt_response = requests.get(srt_endpoint, headers=headers)
    with open("yt.srt", "w") as _file:
        _file.write(srt_response.text)
    
    zip_file = ZipFile('transcription.zip', 'w')
    zip_file.write('yt.txt')
    zip_file.write('yt.srt')
    zip_file.close()






st.sidebar.header('Which video do you want to analyze?')


with st.sidebar.form(key='my_form'):
	#URL = st.text_input('Enter URL of YouTube video:')
	#submit_button = st.form_submit_button(label='Go')
    fileToUpload = st.file_uploader("Upload a custom video..", type=["mp4"])
    submit_button=st.form_submit_button(label='Go')



if submit_button:
    #get_yt(URL)
    if fileToUpload is not None:
        file_details = {"FileName":fileToUpload.name,"FileType":fileToUpload.type}
        with open(os.path.join("",fileToUpload.name),"wb") as f: 
            f.write(fileToUpload.getbuffer())         
        st.success("File has been uploaded!")

    transcribe_yt()

    with open("transcription.zip", "rb") as zip_download:
        btn = st.download_button(
            label="Download ZIP",
            data=zip_download,
            file_name="transcription.zip",
            mime="application/zip"
        )