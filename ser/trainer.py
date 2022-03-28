import os
import glob
import joblib
import numpy as np
import pandas as pd
import librosa
from librosa import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from termcolor import colored
from ser.data import load_data, load_data_TESS


class Trainer:

    def __init__(self, x, y):
        self.model = None
        self.x = x
        self.y = y


    def base_model(self):
        """Initialize the Multi Layer Perceptron Classifier,
        using optimal parameter settings"""

        self.model = MLPClassifier(alpha=0.01,
                              batch_size=32,
                              epsilon=1e-08,
                              hidden_layer_sizes=(300, ),
                              activation = 'logistic',
                              learning_rate='adaptive',
                              max_iter=500)


    def run(self):
        """Train the model using self.x and self.y"""

        self.model.fit(self.x, self.y)


    def evaluate(self, x_test, y_test):
        """Evaluate the model using x_test and y_test, returning baseline
        accuracy, model accuracy score, model precision score (average=macro)"""

        emotion_numbers = pd.DataFrame(data=self.y).value_counts()
        baseline = emotion_numbers.max() / len(self.y)

        y_pred = self.model.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred)
        prec_score = precision_score(y_test, y_pred, average='macro')
        return baseline, acc_score, prec_score


    def predict_probs(self, x_test, y_test, observed_emotions):
        """Calcualte model's predicted probabilities for each emotion,
        returning these as a dataframe.
        Requires x_test, y_test, and observed_emotions (list) as input"""

        emotion_list = observed_emotions
        emotion_list.sort()
        model_pred_prob = pd.DataFrame((self.model.predict_proba(x_test) * 100).round(2),
                                    columns=emotion_list)
        model_pred_prob['prediction'] = self.model.predict(x_test)
        model_pred_prob['actual'] = y_test

        return model_pred_prob


    def confidence_report(self, x_test, y_test, observed_emotions):
        """Print classification report.
        Requires x_test, y_test, and observed_emotions (list) as input"""

        actual = y_test
        predictions = self.model.predict(x_test)
        print(classification_report(actual, predictions, target_names = observed_emotions))


    def confusion_matrix(self, x_test, y_test, observed_emotions):
        """Display confusion matrix.
        Requires x_test, y_test, and observed_emotions (list) as input"""

        cm = confusion_matrix(y_test, self.model.predict(x_test))
        plt.figure(figsize=(12, 10))
        cm = pd.DataFrame(cm,
                        index=[emotion for emotion in observed_emotions],
                        columns=[emotion for emotion in observed_emotions])
        ax = sns.heatmap(cm,
                        linecolor='white',
                        cmap='Blues',
                        linewidth=1,
                        annot=True,
                        fmt='')
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.title('Confusion Matrix', size=20)
        plt.xlabel('Predicted Labels', size=14)
        plt.ylabel('Actual Labels', size=14)
        plt.savefig('Initial_Model_Confusion_Matrix.png')
        plt.show()


    def save_model_locally(self):
        """Save the model into a .joblib format"""

        joblib.dump(self.model, 'MLP_model.joblib')
        print(colored("MLP_model.joblib saved locally", "green"))


if __name__ == "__main__":
    observed_emotions=['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']
    x, y = load_data(
        "/home/iases/code/pankaj-lewagon/ser/raw_data/ravdess_data",
        observed_emotions=observed_emotions)
    # x, y = load_data_TESS('/home/iases/code/pankaj-lewagon/ser/raw_data/TESS')
    # x_train, x_test, y_train, y_test = train_test_split(np.array(x),
    #                                                 y,
    #                                                 test_size=0.2,
    #                                                 shuffle=True,
    #                                                 random_state=9)

    trainer = Trainer(x=x, y=y)
    trainer.base_model()

    trainer.run()
    # baseline, acc_score, prec_score = trainer.evaluate(x_test, y_test)
    # print(f"baseline: {baseline}")
    # print(f"model's accuracy score: {acc_score}")
    # print(f"model's precision score: {prec_score}")

    # print(trainer.predict_probs(x_test, y_test, observed_emotions).head())
    # trainer.confidence_report(x_test, y_test, observed_emotions)
    # trainer.confusion_matrix(x_test, y_test, observed_emotions)

    trainer.save_model_locally()
