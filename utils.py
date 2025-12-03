import pickle

def load_model(path):
    """Load {model, vectorizer} dict"""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj  # returns dict with model + vectorizer


import numpy as np

def predict_sentiment(model_dict, vectorizer, text):
    # Extract model if dict contains both
    if isinstance(model_dict, dict):
        model = model_dict["model"]
    else:
        model = model_dict

    # Vectorize input
    text_vec = vectorizer.transform([text])

    # Predict (model outputs 0,1,2)
    raw_pred = model.predict(text_vec)[0]

    # Normalize to -1, 0, 1
    mapping = {0: -1, 1: 0, 2: 1}
    normalized_pred = mapping.get(int(raw_pred), 0)

    return normalized_pred

