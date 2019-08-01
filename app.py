# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request, jsonify
# scientific computing library for saving, reading, and resizing images
from scipy.misc import imread, imresize
# for matrix math
import numpy as np
# system level operations (like loading files)
import sys
# for reading operating system data
import os
# tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from keras.models import load_model
import pickle
import cv2
import uuid
from keras import backend as K
# initalize our flask app
app = Flask(__name__)

# decoding an image from base64 into raw representation
@app.route('/predict/', methods=['POST'])
def predict():
    K.clear_session()
    model = load_model('./output/smallvggnet.model')   
    lb = pickle.loads(open("./output/smallvggnet_lb.pickle", "rb").read())
    key = str(uuid.uuid4())
    imgData = request.get_data()
    with open('output_file'+key+'.jpg', mode="wb") as new:
        new.write(imgData)
    # encode it into a suitable format
    #convertImage(imgData)
    # read the image into memory
    image = cv2.imread('output_file'+key+'.jpg')
    #output = image.copy()
    image = cv2.resize(image, (64, 64))

    # масштабируем значения пикселей к диапазону [0, 1]
    image = image.astype("float") / 255.0

    
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # загружаем модель и бинаризатор меток
    print("[INFO] loading network and label binarizer...")


    # делаем предсказание на изображении
    preds = model.predict(image)
    print(preds)

    # находим индекс метки класса с наибольшей вероятностью
    # соответствия
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    return jsonify({"class_predicted":'Класс '+str(i+1) + ' - '+label,"probability":str(preds)})
if __name__ == "__main__":
    # run the app locally on the given port
    app.run(host='0.0.0.0', port=3333)
# optional if we want to run in debugging mode
# app.run(debug=True)