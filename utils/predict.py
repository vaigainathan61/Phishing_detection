import joblib
import numpy as np
import tensorflow as tf

def load_models():
    xgb_model = joblib.load('models/xgboost_model.pkl')
    cat_model = joblib.load('models/catboost_model.pkl')
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
    vectorizer = joblib.load('models/vectorizer.pkl')
    return xgb_model, cat_model, lstm_model, vectorizer

def predict(url):
    xgb_model, cat_model, lstm_model, vectorizer = load_models()
    input_vector = vectorizer.transform([url]).toarray()

    xgb_pred = xgb_model.predict_proba(input_vector)[:, 1]
    cat_pred = cat_model.predict_proba(input_vector)[:, 1]
    lstm_pred = lstm_model.predict(input_vector)[0][0]

    # Weighted Ensemble
    weights = [0.4, 0.3, 0.3]  # Assign higher weight to better model
    ensemble_pred = (weights[0] * xgb_pred) + (weights[1] * cat_pred) + (weights[2] * lstm_pred)

    return {
        "XGBoost": xgb_pred[0],
        "CatBoost": cat_pred[0],
        "LSTM": lstm_pred,
        "Ensemble": ensemble_pred[0]
    }
