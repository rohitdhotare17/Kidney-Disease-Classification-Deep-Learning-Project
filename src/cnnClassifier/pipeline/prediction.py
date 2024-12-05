import numpy as np
from keras.models import load_model
from  keras.utils import load_img, img_to_array
   # Correct import
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load the model
        model_path = os.path.join("model", "model.h5")
        model = load_model(model_path)

        # Load and preprocess the image
        imagename = self.filename
        test_image = load_img(imagename, target_size=(224, 224))  # Use the corrected load_img
        test_image = img_to_array(test_image)  # Use the corrected img_to_array
        test_image = np.expand_dims(test_image, axis=0)

        # Predict using the model
        result = np.argmax(model.predict(test_image), axis=1)

        # Interpret and return the prediction
        if result[0] == 1:
            prediction = 'Tumor'
        else:
            prediction = 'Normal'

        return [{"image": prediction}]
