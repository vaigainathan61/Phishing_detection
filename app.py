from flask import Flask, request, render_template
import joblib
import re
import tldextract
import pandas as pd
from flask import Flask, render_template, request
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
import re
import contextlib
import sqlite3
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from create_database import setup_database
from utils1 import login_required, set_session
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import (
    Flask, render_template, 
    request, session, redirect
)


# Load trained model
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
cat_model = joblib.load("models/cat_model.pkl")
tokenizer = joblib.load("models/tokenizer.pkl")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")

app = Flask(__name__)

database = "users.db"
setup_database(name=database)

app.secret_key = 'xpSm7p5bgJY8rNoBjGWiz5yjxM-NEBlW6SIBI62OkLc='

# Feature extraction function
def preprocess_text(url):
    """Preprocess URL for TF-IDF and LSTM models."""
    url_tfidf = vectorizer.transform([url])  # TF-IDF transform
    url_seq = tokenizer.texts_to_sequences([url])  # Tokenize
    url_pad = pad_sequences(url_seq, maxlen=100)  # Pad sequence
    return url_tfidf, url_pad

# Prediction function
def predict_url(url):
    """Predict phishing probability using XGBoost, CatBoost, and LSTM models."""
    url_tfidf, url_pad = preprocess_text(url)

    xgb_pred = xgb_model.predict(url_tfidf)[0]
    cat_pred = cat_model.predict(url_tfidf)[0]
    lstm_pred = (lstm_model.predict(url_pad) > 0.5).astype("int32")[0][0]

    # Weighted Ensemble
    ensemble_pred = round(0.4 * xgb_pred + 0.3 * cat_pred + 0.3 * lstm_pred)

    return {
        "XGBoost_Prediction": int(xgb_pred),
        "CatBoost_Prediction": int(cat_pred),
        "LSTM_Prediction": int(lstm_pred),
        "Final_Ensemble_Prediction": int(ensemble_pred)
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    # Set data to variables
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Attempt to query associated user data
    query = 'select username, password, email from users where username = :username'

    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            account = conn.execute(query, {'username': username}).fetchone()

    if not account: 
        return render_template('login.html', error='Username does not exist')

    # Verify password
    try:
        ph = PasswordHasher()
        ph.verify(account[1], password)
    except VerifyMismatchError:
        return render_template('login.html', error='Incorrect password')

    # Check if password hash needs to be updated
    if ph.check_needs_rehash(account[1]):
        query = 'update set password = :password where username = :username'
        params = {'password': ph.hash(password), 'username': account[0]}

        with contextlib.closing(sqlite3.connect(database)) as conn:
            with conn:
                conn.execute(query, params)

    # Set cookie for user session
    set_session(
        username=account[0], 
        email=account[2], 
        remember_me='remember-me' in request.form
    )
    
    return redirect('/predict_page')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    
    # Store data to variables 
    password = request.form.get('password')
    confirm_password = request.form.get('confirm-password')
    username = request.form.get('username')
    email = request.form.get('email')

    # Verify data
    if len(password) < 8:
        return render_template('register.html', error='Your password must be 8 or more characters')
    if password != confirm_password:
        return render_template('register.html', error='Passwords do not match')
    if not re.match(r'^[a-zA-Z0-9]+$', username):
        return render_template('register.html', error='Username must only be letters and numbers')
    if not 3 < len(username) < 26:
        return render_template('register.html', error='Username must be between 4 and 25 characters')

    query = 'select username from users where username = :username;'
    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            result = conn.execute(query, {'username': username}).fetchone()
    if result:
        return render_template('register.html', error='Username already exists')

    # Create password hash
    pw = PasswordHasher()
    hashed_password = pw.hash(password)

    query = 'insert into users(username, password, email) values (:username, :password, :email);'
    params = {
        'username': username,
        'password': hashed_password,
        'email': email
    }

    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            result = conn.execute(query, params)

    # We can log the user in right away since no email verification
    set_session( username=username, email=email)
    return redirect('/')


@app.route("/predict_page", methods=["POST"])
def predict():
    """Handle user input and display prediction results."""
    url = request.form["url"]
    if not url:
        return render_template("result.html", url=url, prediction="Error: URL missing")

    prediction_result = predict_url(url)
    return render_template(
        "result.html", 
        url=url, 
        xgb=prediction_result["XGBoost_Prediction"],
        cat=prediction_result["CatBoost_Prediction"],
        lstm=prediction_result["LSTM_Prediction"],
        ensemble=prediction_result["Final_Ensemble_Prediction"]
    )

if __name__ == "__main__":
    app.run(debug=True)
