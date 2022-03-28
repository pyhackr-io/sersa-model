from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import requests
import shutil
import joblib
import pandas as pd
import numpy as np
import librosa
import soundfile

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


#Extract features (mfcc, chroma, mel, temp) from a sound file
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


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
@app.get("/")
def index():
    return {"ok docker gcp": True}


@app.post("/predict/")
async def predict(file: UploadFile  = File(...)):

    # Create generic 'ouput' + extension filename to avoid writing too many files on disk
    # As model can handle severeal audio file types we retrieve the extension form provided filename
    # pp=type(file.file)
    filename = 'output.' + str(file.filename)[-3:]
    print(filename)

    with open(filename,'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    observed_emotions = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']

    x_pred_preprocessed = x_pred_preprocessing(filename)
    prediction = return_predict(x_pred_preprocessed)
    predicted_probas =predict_proba(observed_emotions, x_pred_preprocessed).T
    predicted_probas.rename(columns={0 : "probability"}, inplace=True)
    probas=pd.DataFrame.to_dict(predicted_probas)
    probas['emotion']={0:prediction}

    return probas


if __name__ == "__main__":

    url="http://localhost:8000/predict/"
    url2='https://api-btzfftkewq-ew.a.run.app/predict/'
    file_='/Users/pankajpatel/code/pankaj-lewagon/ser/03-01-03-01-02-01-23.wav'
    files={'file': open(file_,'rb') }
    r=requests.post(url, files=files)
    answer = dir(r)
    # print(answer)
    print("good jib")
