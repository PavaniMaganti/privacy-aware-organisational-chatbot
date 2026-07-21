
from pathlib import Path
import joblib


class IntentService:
    """Loads and uses the trained intent-classification model."""

    def __init__(self, model_path: Path):
        self.model_path = model_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Intent model was not found at: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

    def predict(self, text: str) -> str:
        """Predict the intent of a customer question."""

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("Question cannot be empty.")

        prediction = self.model.predict([cleaned_text])[0]

        return str(prediction)
