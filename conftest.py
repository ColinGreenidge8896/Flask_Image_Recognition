"""Configuration for pytest fixtures."""
import pytest
from app import app

@pytest.fixture
def client():
    """A test client for the Flask application."""
    with app.test_client() as client_obj:
        yield client_obj
