from __future__ import division, print_function

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


import os
from os import listdir
from pickle import dump
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense, Bidirectional,Input,Dropout
from tensorflow.python.keras.models import Model
from pickle import load
from keras.utils import to_categorical
import numpy as np
from keras.layers.merge import add
import array as array
from tensorflow.keras.models import load_model
import string

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#define model
filename = 'features.pkl'
all_features = load(open(filename, 'rb'))

# model1 = VGG16()
# model1.layers.pop()
# model1 = Model(inputs=model1.inputs, outputs=model1.layers[-1].output)
# print(model1.summary)

# def extract_features(filename):
#     image = load_img(filename, target_size=(224, 224))
#     image = img_to_array(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     image = preprocess_input(image)
#     feature = model1.predict(image, verbose=0)
#     print('>%s' % len(feature[0]))
#     return feature

model = load_model('model_16.h5')
print(model.summary())

# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')

tokenizer = Tokenizer()
filename = 'tokens.pkl'
tokenizer = load(open(filename, 'rb'))
print(len(tokenizer.word_index) + 1)
max_len = 36

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        imagefile = request.files.get('file', '')
        #get the image feature
        filename1 = secure_filename(imagefile.filename)
        filename2 = 'C:\\Users\\Dell\\Desktop\\flask\\templates\\' + filename1
        print(filename2)
        imagefile.save(filename2)
        name = filename1.split('.')[0]
        photo = all_features[name]
        seed_text = 'startseq'
        for i in range(max_len) :
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            #print(seed_text)
            token_list = pad_sequences([token_list],maxlen=max_len)
            #print(token_list)
            predicted = model.predict([photo,token_list], verbose=0)
            #print(predicted.shape)
            yhat = np.argmax(predicted)
            #print(yhat)
            output_word = ""
            for word,index in tokenizer.word_index.items() :
                if index == yhat :
                    output_word = word
                    break
            if output_word == 'endseq' :
                break
            seed_text += " " + output_word

        return render_template('index.html',filename2=filename1,result=seed_text[9:])
    else :
        return render_template('index.html')
    return None

if __name__ == '__main__':
    app.run(debug=True)