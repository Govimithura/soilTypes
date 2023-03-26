import numpy as np
from flask import Flask, request
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image , ImageOps

app = Flask(__name__)

MODEL_PATH = './soil_types.h5'
loaded_model = tf.keras.models.load_model(MODEL_PATH ,custom_objects={'KerasLayer':hub.KerasLayer})

#----------------------


@app.route('/')
def home():
    return "Hellow World"


@app.route('/predict-soil', methods=['POST'])
def predict_soil():
    #validation steps------------------------------------
    if 'image' not in request.files:
        return 'No image part in the request', 400
    image = request.files['image']
    if image.filename == '':
        return 'No selected image file', 400
    
    #Preparing the Image for the Model--------------------
    image = request.files["image"]
    image_ready = np.array(
        Image.open(image).convert("RGB").resize((448, 448)) # image resizing
    )
    image_ready = image_ready/255 # normalize the image in 0 to 1 range
    img_array = tf.expand_dims(image_ready, 0)
    predictions = loaded_model.predict(img_array)
    class_index = np.argmax(predictions[0])
    preediction_confidence = round(100 * (np.max(predictions[0])), 2)

    class_str = str(class_index)
    confidence_str = str(preediction_confidence)

    #return "Testing"
    return  {"class": class_str , "confidence": confidence_str }


    


if __name__ == '__main__':
    app.run(debug=True)