import joblib
import pandas as pd
from ser.data import extract_features


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


if __name__ == '__main__':
    observed_emotions = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']
    audio_path = 'OAF_back_angry.wav'
    x_pred_preprocessed = x_pred_preprocessing(audio_path)
    prediction = return_predict(x_pred_preprocessed)
    print(prediction)
    predicted_probas = predict_proba(observed_emotions, x_pred_preprocessed)
    print(predicted_probas)
