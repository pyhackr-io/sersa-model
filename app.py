import streamlit as st
import requests
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import soundfile
import librosa
from PIL import Image

st.set_page_config(layout="wide")

# image = Image.open('Sersa_logo.jfif') #logo
# st.image(image, width = 340) #logo width
'''
# SERSA - Speech Emotion Recognizer & Song Advisor
'''



st.sidebar.header('About this project')  #sidebar title
st.sidebar.markdown(
    "**What is SERSA?**  \nSERSA was developed as a deep learning project to identify emotions from speech. SERSA takes a sample of speech as input, analyzes it based on thousands of previous examples of speech and returns the primary emotion it detected in the voice sample. Based on the ouput, SERSA then provides a list of songs scraped from the Spotify API which 'match' the emotion."
)

st.sidebar.markdown(
    "**Why is speech emotion recognition important?**  \nSpeech emotion recognition (SER) is notoriously difficult to do, not just for machines but also for us humans! The applications of SER are varied - from business (improving customer service), to healthcare (telemedicine and supporting people affected by alexithymia) to our personal lives."
)
st.sidebar.markdown(
    '''**What was our approach?** \nUsing a Multilayer Perceptron (MLP) Classifier we were able to train a model on the RAVDESS and TESS datasets and achieve XX percent accuracy. We also tried using CNN and RNN models but they were less effective.'''
)

st.sidebar.markdown(
    "*Sidenotes*:  \nEmotion recognition is an intrinsically subjective task (i.e. what one person considers angry another might consider sad). SERSA was trained on a specific set of voice samples and will therefore extrapolate based on those - thus, you may find SERSA's predictions to be odd at times - that's the nature of the game!"
)

st.subheader(
    ":musical_note: Upload your voice recording here using .wav format")
uploaded_file = st.file_uploader("Select file from your directory")

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes)

    with open("pip.wav", "wb") as file:  #######to be removed and added to package (predict.py) later
        file.write(audio_bytes)  #######



########just for testing until we get the api -- will be removed later#########
def extract_features(file_name, mfcc, chroma, mel, temp):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X,
                                                 sr=sample_rate,
                                                 n_mfcc=40).T,
                            axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft,
                                                         sr=sample_rate).T,
                             axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,
                          axis=0)
            result = np.hstack((result, mel))
        if temp:
            temp = np.mean(librosa.feature.tempogram(y=X, sr=sample_rate).T,
                           axis=0)
            result = np.hstack((result, temp))
    return result

#Load an audio file and transform it
def x_pred_preprocessing(audio_path):
    x_pred_preprocessed = extract_features(audio_path,
                                           mfcc=True,
                                           chroma=False,
                                           mel=True,
                                           temp=True)
    x_pred_preprocessed = x_pred_preprocessed.reshape(1, 552)
    return x_pred_preprocessed


#Predict the emotion
def return_predict(x_pred_preprocessed, model_path='MLP_model.joblib'):
    model = joblib.load(model_path)
    prediction = model.predict(x_pred_preprocessed)
    return prediction[0]


#Return a dataframe giving the predicted probabilities for each emotion in observed_emotions
def predict_proba(observed_emotions, x_pred_preprocessed, model_path='MLP_model.joblib'):
    model = joblib.load(model_path)
    emotion_list = observed_emotions
    emotion_list.sort()
    model_pred_prob = pd.DataFrame((model.predict_proba(x_pred_preprocessed) * 100).round(2),
                                columns=emotion_list)
    return model_pred_prob
######################



# url = ''

button = st.button('click to predict the emotion')

if button:
    # print is visible in the server output, not in the page
    print('button clicked!')

    # response = request.post(url, audio_bytes)
    # response.json()
    # response = {'calm': 63.91, 'happy': 0.00, 'sad': 4.95, 'angry': 30.99, 'fearful':0.14, 'disgust': 0.01}

    observed_emotions = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']
    x_pred_preprocessed = x_pred_preprocessing('pip.wav')
    prediction = return_predict(x_pred_preprocessed)
    st.write(prediction)
    predicted_probas = predict_proba(observed_emotions, x_pred_preprocessed)
    hpp = predicted_probas.assign(hack='').set_index('hack')
    st.write(hpp)
    st.bar_chart(predicted_probas)
