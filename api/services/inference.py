import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer


class Inference:
    def __init__(self, model_path: str):
        model_dir = Path(model_path)

        self.encoder = SentenceTransformer(
            str(model_dir / "sentence_transformer.model")
        )
        self.classifier = joblib.load(model_dir / "classifier.joblib")

        self.id_to_label = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }

    def predict(self, text: str) -> str:
        embedding = self.encoder.encode([text], convert_to_numpy=True)
        pred_id = int(self.classifier.predict(embedding)[0])
        return self.id_to_label[pred_id]
