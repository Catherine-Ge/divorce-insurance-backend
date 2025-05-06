import os
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)

# Get project root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
model_dir = os.path.join(root_dir, "insurance_design")

# Load model and encoder
model = joblib.load(os.path.join(model_dir, "logistic_model.pkl"))
encoder = joblib.load(os.path.join(model_dir, "onehot_encoder.pkl"))

import smtplib
from email.mime.text import MIMEText

def send_email(to_address, subject, body):
    from_address = "6556catherine@gmail.com"
    password = os.environ.get("GMAIL_APP_PASSWORD")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_address
    msg["To"] = to_address

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_address, password)
        server.send_message(msg)

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    if "email" in df.columns:
        df = df.drop(columns=["email"])  #remove the email before encoding
    X = encoder.transform(df)
    proba = model.predict_proba(X)[0][1]

    base_premium = 500
    premium = base_premium + proba * 1000
    payout = 10000 if proba > 0.5 else 5000

    if "email" in data:
         quote_text = f"""Thank you for using Divorce Insurance!
        Estimated Divorce Risk: {round(proba * 100, 1)}%
        Premium: ${round(premium, 2)}
        Payout on Divorce: ${payout}
    """
    send_email(data["email"], "Your Divorce Insurance Quote", quote_text)

    return jsonify({
        "divorce_risk": round(proba, 3),
        "premium": round(premium, 2),
        "payout": payout
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)