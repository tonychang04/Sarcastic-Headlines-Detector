import string

from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget

import pickle

from nltk.corpus import stopwords


def text_process_for_ML(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    # print(nopunc)

    # print(no_stop_words)
    return [word for word in nopunc.split() if word.lower() not in
            stopwords.words('english')]

class MyGrid(Widget):

    def predict(self):
        filenameBOW = 'Bag-Of-Words'
        filenameSVC = 'Support Vector Machine'

        print('nice')
        BOW_model = pickle.load(open(filenameBOW, 'rb'))

        SVC_model = pickle.load(open(filenameSVC, 'rb'))

        input_string = [self.input.text]

        process_string = BOW_model.transform(input_string)
        print(input_string)
        print(process_string)
        prediction = SVC_model.predict(process_string)
        prediction_array = SVC_model.predict_proba(process_string)
        print(prediction)
        print(prediction_array)






