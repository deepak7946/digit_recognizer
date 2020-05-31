import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dropout, Flatten, Conv2D, Dense
import datetime
import numpy as np
from imageio import imread, imsave
from skimage.transform import resize
#import tensorflow as tf

img_size = (28,28)
num_classes = 10

class digit_recog_model:
    def __init__(self, img_dim=img_size, in_ch=1, num_classes=10):
        self.model = None
        self.img_width = img_dim[0]
        self.img_height = img_dim[1]
        self.in_ch = in_ch
        self.num_classes = num_classes
        return
    
    def data_prep(self, data, train=False, scale=True):
        if train:
            labels = keras.utils.to_categorical(data.label, num_classes=self.num_classes)
            num_images = data.shape[0]
            x_array = data.values[:,1:]
            out_x = x_array.reshape(num_images, self.img_width, self.img_height, self.in_ch)
            out_x = out_x/255
        else:
            labels = None
            num_images = data.shape[0]
            x_array = data
            out_x = x_array.reshape(num_images, self.img_width, self.img_height, self.in_ch)
            if scale:
                out_x = out_x/255
        return out_x, labels
    
    def create_model(self, stride=1, verbose=True):
        self.model = Sequential()
        self.model.add(Conv2D(12, kernel_size=(3,3), strides=stride, activation='relu', input_shape=(self.img_width, self.img_height, self.in_ch)))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(16, kernel_size=(3,3), strides=stride, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(20, kernel_size=(3,3), strides=stride, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        if verbose:
            self.model.summary()
        self.model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        return
    
    def train_model(self, train_set, plot=True, save=False):
        print("Training Model")
        x,y = self.data_prep(train_set, train=True)
        history = self.model.fit(x, y, batch_size=50, epochs=10, validation_split=0.2, verbose=False)
        print("Model training Complete")
        print("Final accruacy:\nTraining: {}\nValidation: {}" .format(history.history['accuracy'][-1], history.history['val_accuracy'][-1]))
        if plot:
            self.plot_train(history)
        if save:
            self.save_model()
        return
    
    def save_model(self):
        model_json = self.model.to_json()
        ts = str(datetime.datetime.now().timestamp()).split(".")[0]
        model_file = "model.json"
        print("Writing model to {}" .format(model_file))
        path = ("model/{}" .format(model_file))
        with open(path, "w") as f:
            f.write(model_json)
        weight_file = "model_coeff.h5"
        path = ("model/{}" .format(weight_file))
        print("Writing model weights to {}" .format(weight_file))
        self.model.save_weights(path)
        print("Saved the model")
        return
    
    def load_model(self, model_file, model_coeff):
        with open(model_file, 'r') as json_file:
            model_json = json_file.read()
        self.model = model_from_json(model_json)
        self.model.load_weights(model_coeff)
        self.model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print("Loaded model from disk")
        
    def plot_train(self, history):
        try:
            from matplotlib import pyplot
        except Exception:
            print("Plotting required matplotlib. 'pip install matplotlib==3.2.1'")
            return
        pyplot.plot(history.history['accuracy'], label='train')
        pyplot.plot(history.history['val_accuracy'], label='validation')
        pyplot.legend()
        pyplot.show()
        return
    
    def evaluate_model(self, data):
        X, y = self.data_prep(data, train=True)
        _, acc_score = self.model.evaluate(X, y, verbose=False)
        print("accuracy = {}" .format(acc_score))
    
    def predict_digit(self, img, scale=False):
        img, _ = self.data_prep(img, scale=False)
        num_prob = self.model.predict(img)
        print (num_prob)
        num = np.argmax(num_prob)
        print(num)
        return num

def train_model(train_data):
    digit_model = digit_recog_model()
    digit_model.create_model()
    digit_model.train_model(train_data, plot=False, save=True)
    return digit_model

def load_model(model_file, model_coeff):
    digit_model = digit_recog_model()
    digit_model.load_model(model_file, model_coeff)
    return digit_model
    
if __name__ == "__main__":
    train_data = pd.read_csv("train.csv")
    img = imread('output.png', pilmode='L')
    print(train_data.iloc[0,:])
    img = resize(img, (28, 28))
    print(img)

