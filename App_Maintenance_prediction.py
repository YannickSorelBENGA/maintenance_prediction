import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report,f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

# load the model
model_package = joblib.load("Maintenance_prediction_model.pkl")

model = model_package["model"]
threshold = model_package["threshold"]
feature_names = model_package["feature_names"]
model_name = model_package["model_name"]

st.set_page_config(
    page_title="Predictive Maintenance App",
    page_icon="🛠️",
    layout="centered"
)


st.title("🛠️ Predictive Maintenance App")
st.write("This application predicts whether a machine is likely to fail using the final Random Forest model.")

def reset_inputs():
    for feature in feature_names:
        st.session_state[f"input_{feature}"] = 0.0


if st.button("🔄 Refresh / Reset"):
    reset_inputs()
    st.rerun()

st.subheader("Enter machine information")

input_data = {}

for feature in feature_names:
    input_data[feature] = st.number_input(
        label=feature,
        value=0.0,
        step=1.0,
        key=f"input_{feature}"
    )

# Convert input into dataframe

input_df = pd.DataFrame([input_data])

st.subheader("Input Data")
st.dataframe(input_df)

if st.button("Predict Machine Failure"):

    # Get probability of failure
    failure_probability = model.predict_proba(input_df)[0][1]

    # Apply threshold
    prediction = 1 if failure_probability >= threshold else 0

    st.subheader("Prediction Result")

    st.write(f"Model used: **{model_name}**")
    st.write(f"Decision threshold: **{threshold:.2f}**")
    st.write(f"Failure probability: **{failure_probability:.2%}**")

    if prediction == 1:
        st.error(" High risk of machine failure detected.")
        st.write("Recommended action: Schedule maintenance inspection.")
    else:
        st.success(" Low risk of machine failure.")
        st.write("Recommended action: Continue normal monitoring.")


