from pathlib import Path
import pytest
from api.services.inference import Inference


@pytest.fixture(scope="session")
def model():
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"
    return Inference(str(model_dir))


def test_model_loads_without_errors(model):
    assert model is not None
    assert model.encoder is not None
    assert model.classifier is not None


@pytest.mark.parametrize(
    "text",
    [
        "I love this movie.",
        "The movie starts at 20:00.",
        "This movie was terrible.",
    ],
)
def test_inference_works_for_sample_strings(model, text):
    prediction = model.predict(text)

    assert isinstance(prediction, str)
    assert prediction in {"negative", "neutral", "positive"}
