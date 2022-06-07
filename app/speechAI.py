import os
from django.conf import settings
import pickle
import random
from colorama import Fore, Style, Back
import speech_recognition as sr
import pyttsx3
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
colorama.init

STATIC_DIR = settings.STATIC_DIR

with open('intents.json') as file:
    data = json.load(file)

r = sr.Recognizer()

# Function to convert text to
# speech


def SpeakText(command):

    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


# Loop infinitely for user to
# speak
model = keras.models.load_model(os.path.join(STATIC_DIR, 'chat_model'))
with open(os.path.join(STATIC_DIR, 'tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)
with open(os.path.join(STATIC_DIR, 'lbl_encoder.pickle'), 'rb') as ecn:
    lbl_encoder = pickle.load(ecn)
max_len = 20

while(1):

    # Exception handling to handle
    # exceptions at the runtime
    try:

        # use the microphone as source for input.
        with sr.Microphone() as source2:

            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=0.2)

            # listens for the user's input
            audio2 = r.listen(source2)

            # Using ggogle to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            #print(Fore.LIGHTBLUE_EX + "User: "+ Style.RESET_ALL,end="")
            #inp = input()
            if MyText.lower() == "quit":
                break

            result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([MyText]),
                                                                              truncating='post', maxlen=max_len))

            tag = lbl_encoder.inverse_transform([np.argmax(result)])
            for i in data['intents']:
                if i['tag'] == tag:
                    print(Fore.GREEN + "chatBot: " + Style.RESET_ALL,
                          np.random.choice(i['responses']))
                    print("Did you say "+np.random.choice(i['responses']))
                    SpeakText(np.random.choice(i['responses']))

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("unknown error occured")
