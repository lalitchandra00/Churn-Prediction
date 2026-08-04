import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle


model = tf.keras.models.load_model('model.h5')

with open('le.pkl', 'rb') as file:
    le = pickle.load(file)

with open('ohe.pkl', 'rb') as file:
    ohe = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Customer Churn Prediction")
st.markdown("Fill in the customer details below and click **Predict** to find out if the customer is likely to churn.")

st.header("👤 Enter Customer Details")

with st.form("customer_form"):
    c1, c2 = st.columns(2)

    with c1:
        geography = st.selectbox("🌍 Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        has_cr_card = st.radio("💳 Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.radio("✅ Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        num_of_products = st.slider("🛒 Number of Products", min_value=1, max_value=4, value=2)

    with c2:
        credit_score = st.slider("🎯 Credit Score", min_value=300, max_value=900, value=600)
        age = st.slider("🎂 Age", min_value=18, max_value=100, value=40)
        tenure = st.slider("📅 Tenure (years)", min_value=0, max_value=10, value=5)
        balance = st.number_input("💰 Balance", min_value=0.0, value=60000.0, step=1000.0)
        estimated_salary = st.number_input("💵 Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

    submitted = st.form_submit_button("🚀 Predict Churn", use_container_width=True)

if submitted:
    input_data = {
        'HasCrCard': has_cr_card,
        'CreditScore': credit_score,
        'Gender': gender,
        'Geography': geography,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'PrdtcCnts': num_of_products,
        'ActivedMmbr': is_active_member,
        'EstimatedSalary': estimated_salary,
    }

    geo_encoded = ohe.transform([[input_data['Geography']]]).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=ohe.get_feature_names_out(['Geography'])
    )

    input_df = pd.DataFrame([input_data])
    input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)
    input_df['Gender'] = le.transform([input_data['Gender']])
    input_df.drop(columns=['Geography'], inplace=True)

    input_df_scaled = scaler.transform(input_df)

    prediction = model.predict(input_df_scaled)
    prediction_value = prediction[0][0]

    st.subheader("📈 Prediction Result")

    st.progress(float(prediction_value))
    st.write(f"**Churn Probability: {prediction_value:.2%}**")

    if prediction_value > 0.5:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is not likely to churn.")
