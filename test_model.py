"""Unit Tests for Image Recognition Model"""
import pytest
import numpy as np
from keras.models import load_model
from model import preprocess_img, predict_result  # Adjust based on your structure

# Adjust if your model has a different number of output classes
NUM_CLASSES = 10

# Load the model before tests run
@pytest.fixture(scope="module")
def model():
    """Load the model once for all tests."""
    model_test = load_model("digit_model.h5")  # Adjust path as needed
    return model_test


# Basic Tests
def test_preprocess_img():
    """Test the preprocess_img function."""
    img_path = "test_images/2/Sign 2 (97).jpeg"
    processed_img = preprocess_img(img_path)

    # Check that the output shape is as expected
    assert processed_img.shape == (1, 224, 224, 3)

    # Check that values are normalized (between 0 and 1)
    assert np.min(processed_img) >= 0 and np.max(processed_img) <= 1


def test_predict_result():
    """Test the predict_result function."""
    img_path = "test_images/4/Sign 4 (92).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)

    # Print the prediction for debugging
    print(f"Prediction: {prediction} (Type: {type(prediction)})")

    # Check that the prediction is an integer (convert if necessary)
    assert isinstance(prediction, (int, np.integer)), "Prediction should be an integer class index"


def test_preprocess_img_dtype_and_finite():
    """Basic: preprocessed image should be float32 with only finite values."""
    img_path = "test_images/4/Sign 4 (92).jpeg"
    processed_img = preprocess_img(img_path)

    # dtype must be float32 and all numbers finite (no NaN/inf)
    assert processed_img.dtype == np.float32, "Preprocessed image must be float32"
    assert np.isfinite(processed_img).all(), "Preprocessed image must not contain NaN/inf values"


def test_predict_class_index_in_bounds():
    """Basic: predicted class index should be within [0, NUM_CLASSES-1]."""
    img_path = "test_images/2/Sign 2 (97).jpeg"
    processed_img = preprocess_img(img_path)
    pred = predict_result(processed_img)

    assert isinstance(pred, (int, np.integer)), "Prediction must be an integer"
    assert 0 <= int(pred) < NUM_CLASSES, f"Prediction must be in [0, {NUM_CLASSES-1}]"


# Advanced Tests
def test_invalid_image_path():
    """Test preprocess_img with an invalid image path."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("invalid/path/to/image.jpeg")


def test_image_shape_on_prediction():
    """Test the prediction output shape."""
    img_path = "test_images/5/Sign 5 (86).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Ensure that the prediction output is an integer
    prediction = predict_result(processed_img)
    assert isinstance(prediction, (int, np.integer)), "The prediction should be an integer"


def test_model_predictions_consistency():
    """Test that predictions for the same input are consistent."""
    img_path = "test_images/7/Sign 7 (54).jpeg"
    processed_img = preprocess_img(img_path)

    # Make multiple predictions
    predictions = [predict_result(processed_img) for _ in range(5)]

    # Check that all predictions are the same
    assert all(p == predictions[0] for p in predictions)


def test_batch_predictions_multiple_images_type_and_bounds():
    """Advanced: run over several images; every prediction is an int within bounds."""
    candidates = [
        "test_images/2/Sign 2 (97).jpeg",
        "test_images/4/Sign 4 (92).jpeg",
        "test_images/5/Sign 5 (86).jpeg",
        "test_images/7/Sign 7 (54).jpeg",
    ]
    preds = []
    for path in candidates:
        x = preprocess_img(path)
        preds.append(predict_result(x))

    # same count, all ints, all within class range
    assert len(preds) == len(candidates)
    assert all(isinstance(y, (int, np.integer)) for y in preds), "All predictions must be integers"
    assert all(0 <= int(y) < NUM_CLASSES for y in preds), "All predictions must be within class bounds"


def test_preprocess_img_rejects_non_string_path():
    """Advanced: passing a non-string path should raise a clear error."""
    with pytest.raises((TypeError, ValueError)):
        preprocess_img(None)  # unsupported type on purpose