# digit_recognizer

## Description
The app is a handwritten digit rezcognizer.

## Usage
- Run the webapi.py and login to the go to the url 127.0.0.1:8000
- Draw the digit on the black screen and click on predict
- Click clear to clear the canvas and the prediction output bar

## Model
The digit recognizer uses a CNN (using Keras) in the backend to recognize the digits. The prediction works best with linewidth of ~25 on canvas
Training data is from kaggle (https://www.kaggle.com/c/digit-recognizer). 
Flask is used to create the APIs for the app.

## Reference
javascript code reference (https://www.pytorials.com/deploy-keras-model-to-production-using-flask/)