import speech_recognition as sr
import pyttsx3 
from translate import Translator
from gtts import gTTS
from playsound import playsound
import json
import numpy as np 
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
import colorama
colorama.init
from colorama import Fore, Style, Back
import random
import pickle
import os
from pathlib import Path
from django.shortcuts import render,redirect

with open('E:\\jerin\\Voice-Assistant\\NUBI\\static\\intents.json') as file:
    data = json.load(file)
df = pd.read_csv('E:\\jerin\\Voice-Assistant\\NUBI\\static\\books.csv', error_bad_lines = False)
def num_into_obj(x):
    if x>=0 and x<=1:
        return 'between 0 and 1'
    elif x>1 and x<=2:
        return 'between 1 and 2'
    elif x>2 and x<=3:
        return 'between 2 and 3'
    elif x>3 and x<=4:
        return 'between 3 and 4'
    else:
        return 'between 4 and 5'
    
df['rating_obj'] = df['average_rating'].apply(num_into_obj)
rating_df = pd.get_dummies(df['rating_obj'])
language_df = pd.get_dummies(df['language_code'])
features = pd.concat([rating_df, language_df, df['average_rating'], df['ratings_count'],df['title']], axis=1)
features.set_index('title', inplace=True)
min_max_scaler = MinMaxScaler()
features_scaled = min_max_scaler.fit_transform(features)
model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric='euclidean')
model.fit(features_scaled)
dist, idlist = model.kneighbors(features_scaled)
r = sr.Recognizer() 
def SpeakText(command):
      
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()


def index2(request):
    return render(request, 'index.html')


def chat(request):
    
    model = keras.models.load_model('E:\\jerin\\Voice-Assistant\\NUBI\\static\\voice_assistant')
    with open('E:\\jerin\\Voice-Assistant\\NUBI\\static\\tokenizer.pickle','rb') as handle:
        tokenizer = pickle.load(handle)
    with open('E:\\jerin\\Voice-Assistant\\NUBI\\static\\lbl_encoder.pickle','rb') as ecn:
        lbl_encoder = pickle.load(ecn)
    max_len = 20
    
    while(1):
        try:
            
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                translator=Translator(from_lang='en',to_lang="ml")
                print("User: "+ MyText)
                if MyText.lower() == "bye":
                    break
                if MyText.lower() == "book search":
                    text="Tell book name you like"
                    translation=translator.translate(text)
                    print("Tell book name you like")
                    print(Fore.GREEN + "NUBI: "+translation)
                    mal=gTTS(translation,lang="ml")
                    mal.save("voice5.mp3")
                    playsound("voice5.mp3")
                    os.remove("voice5.mp3")
                    #print("Tell book name you like")
                    #SpeakText("Tell book name you like")
                    while(1):
                        try:
                            with sr.Microphone() as source2:
                                r.adjust_for_ambient_noise(source2, duration=0.2)
                                audio2 = r.listen(source2)
                                MyBook = r.recognize_google(audio2)
                                MyBooks = MyBook.lower()
                                book=MyBooks.title()
                                print(book)
                                
                                def BookRecommender(book):
                                    book_list_name = []
                                    book_id = df[df['title'] == book].index
                                    book_id = book_id[0]
                                    for newid in idlist[book_id]:
                                        book_list_name.append(df.loc[newid].title)
                                        SpeakText(df.loc[newid].title)
                                        print(df.loc[newid].title)
                                    return
                                if MyText.lower() == "thank you":
                                    break
                                print(Fore.GREEN + "NUBI: "+"I would prefer that you read.")
                                SpeakText("I would prefer that you read")
                                BookRecommender(book)
                        except sr.RequestError as e:
                            print("Could not request results; {0}".format(e))
                            
            result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([MyText]),
                                                                           truncating='post',maxlen=max_len))
            
            tag = lbl_encoder.inverse_transform([np.argmax(result)])
            for i in data['intents']:
                if i['tag']==tag:
                    print(Fore.GREEN + "NUBI: "+ Style.RESET_ALL,np.random.choice(i['responses']))
                    translation=translator.translate(np.random.choice(i['responses']))
                    print("NUBI:"+translation)
                    res="NUBI: "+translation+""
                    datas={'res':translation}                   
                    mal=gTTS(translation,lang="ml")
                    mal.save("voice.mp3")
                    playsound("voice.mp3")
                    os.remove("voice.mp3")
                    SpeakText(np.random.choice(i['responses']))
                    render(request, 'index.html',context=datas)
                    
        except sr.RequestError as e:
             print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            translator=Translator(from_lang='en',to_lang="ml")
            text="I'm sorry, but I'm unable to hear you."
            translation=translator.translate(text)
            print("NUBI:"+translation)
            res="NUBI: "+translation+""
            datas={'res':res}
            mal=gTTS(translation,lang="ml")
            mal.save("voice2.mp3")
            playsound("voice2.mp3")
            os.remove("voice2.mp3")
            render(request, 'index.html',context=datas)
           
                            
        