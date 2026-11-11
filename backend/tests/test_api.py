from io import BytesIO
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


def create_test_image() -> bytes:
    image = Image.new("RGB", (64, 64), color="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_analyze_endpoint():
    mock_classifier = Mock()
    mock_classifier.predict.return_value = {
        "category": "Apparel",
        "confidence": 0.98,
    }

    mock_retriever = Mock()
    mock_retriever.search.return_value = [
        {
            "id": 123,
            "name": "Test Shirt",
            "category": "Apparel",
            "article_type": "Shirts",
            "similarity": 0.91,
        }
    ]

    # Save previous state to avoid leaking mocks between tests
    previous_classifier = getattr(
        app.state, "classifier", None
    )
    previous_retriever = getattr(
        app.state, "retriever", None
    )
    previous_ready = getattr(
        app.state, "models_ready", False
    )

    try:
        app.state.classifier = mock_classifier
        app.state.retriever = mock_retriever
        app.state.models_ready = True

        # No context manager: lifespan is not started
        client = TestClient(app)

        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.png",
                    create_test_image(),
                    "image/png",
                )
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["classification"]["category"] == "Apparel"
        assert data["classification"]["confidence"] == 0.98

        assert len(data["similar_products"]) == 1
        assert data["similar_products"][0]["id"] == 123

        mock_classifier.predict.assert_called_once()
        mock_retriever.search.assert_called_once()

    finally:
        app.state.classifier = previous_classifier
        app.state.retriever = previous_retriever
        app.state.models_ready = previous_ready
