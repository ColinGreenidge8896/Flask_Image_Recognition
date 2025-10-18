"""Integration Test: No File Uploaded Scenario"""
import pytest
from app import app


@pytest.fixture
def client():
    """Fixture for the Flask test client."""
    with app.test_client() as client_obj:
        yield client_obj

def test_missing_file(client_obj):
    """Test the prediction route with a missing file."""
    response = client_obj.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data  # Check if the error message is displayed
