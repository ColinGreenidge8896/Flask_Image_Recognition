"""Convert a Keras .h5 model to TensorFlow SavedModel format."""

from keras.models import load_model

model = load_model("digit_model.h5", compile=False)

model.save("digit_model_saved")
