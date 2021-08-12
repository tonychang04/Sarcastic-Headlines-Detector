import string

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


        print("works!")

