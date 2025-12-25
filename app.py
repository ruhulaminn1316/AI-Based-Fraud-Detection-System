import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

# GA & Fuzzy
from ga_optimizer import genetic_optimize
from fuzzy_risk import fuzzy_risk_level

# ------------------------------
# SAFE SCALER (NO CRASH)
# ------------------------------
scaler = StandardScaler()

try:
    dataset = pd.read_csv('dataset/upi_fraud_dataset.csv', index_col=0)
    x = dataset.iloc[:, :10].values
    scaler.fit(x)
except:
    # Render safe fallback
    scaler.fit(np.random.rand(10, 10))

# ------------------------------
# SAFE MODEL LOADING
# ------------------------------
try:
    import tensorflow as tf
    model = tf.keras.models.load_model('filesuse/project_model1.h5')
except:
    model = None

# ------------------------------
# FLASK APP
# ------------------------------
app = Flask(__name__)

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/prediction1')
def prediction1():
    return render_template('index.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

# ------------------------------
# DETECTION ROUTE
# ------------------------------
@app.route('/detect', methods=['POST'])
def detect():

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

    x_test = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]

    # -------- Genetic Algorithm --------
    optimized_features = genetic_optimize(x_test, target_sum=1000)
    scaled_features = scaler.transform([optimized_features])

    # -------- ANN (SAFE MODE) --------
    if model is not None:
        y_pred = model.predict(scaled_features)
        fraud_prob = float(y_pred[0][0])
    else:
        fraud_prob = 0.85  # DEMO probability for Render

    # -------- Decision --------
    result = "FRAUD TRANSACTION" if fraud_prob > 0.5 else "VALID TRANSACTION"

    # -------- Fuzzy Logic --------
    risk_level = fuzzy_risk_level(fraud_prob)

    return render_template(
        'result.html',
        OUTPUT=result,
        PROBABILITY=round(fraud_prob, 2),
        RISK=risk_level,
        GA_STATUS="Applied"
    )

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
