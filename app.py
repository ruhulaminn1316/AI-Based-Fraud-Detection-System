import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from flask import Flask, request, render_template

# 🔹 NEW: GA & Fuzzy imports
from ga_optimizer import genetic_optimize
from fuzzy_risk import fuzzy_risk_level


dataset = pd.read_csv('dataset/upi_fraud_dataset.csv', index_col=0)

x = dataset.iloc[:, :10].values
y = dataset.iloc[:, 10].values

scaler = StandardScaler()
scaler.fit_transform(x)

model = tf.keras.models.load_model('filesuse/project_model1.h5')

app = Flask(__name__)

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df)

@app.route('/prediction1', methods=['GET'])
def prediction1():
    return render_template('index.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')


# =========================================================
# 🔥 PREDICTION ROUTE (GA + ANN + FUZZY LOGIC HERE)
# =========================================================
@app.route('/detect', methods=['POST'])
def detect():

    # -------- Existing feature extraction (UNCHANGED) --------
    trans_datetime = pd.to_datetime(request.form.get("trans_datetime"))
    v1 = trans_datetime.hour
    v2 = trans_datetime.day
    v3 = trans_datetime.month
    v4 = trans_datetime.year
    v5 = int(request.form.get("category"))
    v6 = float(request.form.get("card_number"))

    dob = pd.to_datetime(request.form.get("dob"))
    v7 = np.round((trans_datetime - dob).days / 365.25)

    v8 = float(request.form.get("trans_amount"))
    v9 = int(request.form.get("state"))
    v10 = int(request.form.get("zip"))

    # 🔹 Raw input features
    x_test = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]

    # =========================================================
    # 🧬 STEP-1: Genetic Algorithm (BEFORE ANN)
    # =========================================================
    optimized_features = genetic_optimize(x_test, target_sum=1000)

    # Scaling optimized features
    scaled_features = scaler.transform([optimized_features])

    # =========================================================
    # 🤖 STEP-2: ANN Prediction
    # =========================================================
    y_pred = model.predict(scaled_features)
    fraud_prob = y_pred[0][0]

    if fraud_prob <= 0.5:
        result = "VALID TRANSACTION"
    else:
        result = "FRAUD TRANSACTION"

    # =========================================================
    # 🌡️ STEP-3: Fuzzy Logic (AFTER ANN)
    # =========================================================
    risk_level = fuzzy_risk_level(fraud_prob)

    # -------- Send output to UI --------
    return render_template(
        'result.html',
        OUTPUT=result,
        PROBABILITY=round(fraud_prob, 2),
        RISK=risk_level,
        GA_STATUS="Applied"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
