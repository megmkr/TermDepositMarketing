import pickle
import pandas as pd
from sklearn.metrics import f1_score

def load_model(path):

    with open(path, "rb") as f:
        model = pickle.load(f)

    if isinstance(model, dict):
        model = model["model"]

    return model


def predict(model_path, X):
    model = load_model(model_path)
    return model.predict(X)

def evaluate_models(X_test, y_test, models):
    results = []

    for name, model_path in models.items():
        y_pred = predict(model_path, X_test)

        results.append({
            "model": name,
            "f1_score": f1_score(y_test, y_pred)
        })

    return pd.DataFrame(results)
