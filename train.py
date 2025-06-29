import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Load dataset
df = pd.read_csv("dataset/phishing_data.csv", encoding="ISO-8859-1")  # Ensure dataset.csv has 'url' and 'label' (values: 'good' or 'bad')

# 🔹 Convert labels: "good" → 0, "bad" → 1
df["Label"] = df["Label"].map({"good": 0, "bad": 1})

# Text preprocessing
X = df["URL"]
y = df["Label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1️⃣ **Train TF-IDF + XGBoost**
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train_tfidf, y_train)

# 2️⃣ **Train TF-IDF + CatBoost**
cat_model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, verbose=0)
cat_model.fit(X_train_tfidf, y_train)

# 3️⃣ **Train LSTM Model**
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=100, padding="post")

vocab_size = 5000
embedding_dim = 64

lstm_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=100),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
lstm_model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test), epochs=5, batch_size=32)

# Save models
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(cat_model, "models/cat_model.pkl")
joblib.dump(tokenizer, "models/tokenizer.pkl")
lstm_model.save("models/lstm_model.h5")

print("✅ Models trained and saved successfully!")
