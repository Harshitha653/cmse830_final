import pickle
import numpy as np


def load_model(path: str):
    """
    Load a model bundle from disk.

    The bundle is expected to be a dict with keys:
      - 'model': trained classifier
      - 'vectorizer': fitted vectorizer
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def predict_sentiment(model_bundle, vectorizer, text: str) -> int:
    """
    Predict sentiment for a single input string.

    model_bundle can be:
      - dict with key 'model'
      - or a sklearn-style estimator directly

    Returns:
      -1 (negative), 0 (neutral), or 1 (positive)
    """
    if isinstance(model_bundle, dict):
        model = model_bundle.get("model")
    else:
        model = model_bundle

    text_vec = vectorizer.transform([text])
    raw_pred = model.predict(text_vec)[0]

    # Original model labels: 0=negative, 1=neutral, 2=positive
    mapping = {0: -1, 1: 0, 2: 1}
    return mapping.get(int(raw_pred), 0)
