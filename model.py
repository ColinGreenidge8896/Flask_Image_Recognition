"""Model loading and prediction functions for image recognition."""
from keras.utils import img_to_array
from keras.layers import TFSMLayer
import numpy as np
from PIL import Image

# Load the TensorFlow SavedModel using TFSMLayer
model = TFSMLayer("digit_model_new", call_endpoint="serving_default")


def preprocess_img(img_path):
    """Preprocess the input image for prediction."""
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


def predict_result(img):
    """Predict the class of the input image using the loaded model."""
    # img should already be preprocessed as a batch (shape: [1, height, width, channels])
    pred = model(img)          # returns a tensor
    pred = np.argmax(pred, axis=-1)  # get predicted class
    return pred[0]
