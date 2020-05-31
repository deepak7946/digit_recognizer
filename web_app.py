from flask import Flask, render_template, request
from imageio import imread, imsave
from skimage.transform import resize
from digit_recognition_using_cnn import load_model
import re
import sys 
import os
import base64


global model
model = load_model("model/model.json", "model/model_coeff.h5")
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

def convert_imgdata(img_data):
    img_str = re.search(r'base64,(.*)', str(img_data)).group(1)
    with open('output.png', 'wb') as f:
        f.write(base64.b64decode(img_str))

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    img_data = request.get_data()
    convert_imgdata(img_data)
    img = imread('output.png', pilmode='L')
    img = resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)
    response = str(model.predict_digit(img))
    return response


if __name__ == "__main__":
    app.run(debug=True, port=8000)