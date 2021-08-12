from app.detector_app import DetectorApp
import pandas as pd
import numpy as np
import string

import pickle

# Getting weights of words starts here
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #support vector machine

import nltk
from nltk.corpus import stopwords



def text_process_for_ML(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    # print(nopunc)

    # print(no_stop_words)
    return [word for word in nopunc.split() if word.lower() not in
            stopwords.words('english')]


if __name__ == '__main__':
    no_model = False
    if (no_model):
        df1 = pd.read_json('datasets/Sarcasm_Headlines_Dataset.json', lines=True)
        df2 = pd.read_json('datasets/Sarcasm_Headlines_Dataset_v2.json', lines=True)
        frames = [df1, df2]
        df = pd.concat(frames)  # merged two json files into 1 dataframe file
        print(df.head(10))

        X_train, X_test, Y_train, Y_test = train_test_split(df['headline'], df['is_sarcastic'], test_size=0.3)
        # bag of words = BOW

        filenameBOW = 'Bag-Of-Words'
        BOW_transform = CountVectorizer(analyzer=text_process_for_ML)
        BOW_transform.fit(X_train)
        vector = BOW_transform.transform(X_train)

        pickle.dump(BOW_transform, open(filenameBOW, 'wb'))

        filenameSVC = 'Support Vector Machine'
        support_vector_machine = SVC(probability=True)
        support_vector_machine.fit(vector, Y_train)

        pickle.dump(support_vector_machine, open(filenameSVC, 'wb'))





    detectorApp = DetectorApp()
    detectorApp.run()

if __name__ == 'other':
    print("nice")


