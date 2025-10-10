from keras.models import load_model

# Load the old .h5 model
model = load_model("digit_model.h5", compile=False)

# Save it in the new SavedModel format
model.save("digit_model_saved")  # This creates a folder
