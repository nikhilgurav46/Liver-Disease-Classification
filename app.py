import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("rf_smote_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="Liver Disease Prediction", layout="centered")

st.title("🩺 Liver Disease Prediction App")
st.markdown("Predict liver disease category using a **Random Forest model trained with SMOTE**.")

st.header("📋 Enter Patient Details")

def user_input_features():
    age = st.number_input("Age", min_value=1, max_value=100, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    albumin = st.number_input("Albumin", value=4.0)
    alkaline_phosphatase = st.number_input("Alkaline Phosphatase", value=100.0)
    alanine_aminotransferase = st.number_input("Alanine Aminotransferase (ALT)", value=30.0)
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (AST)", value=30.0)
    bilirubin = st.number_input("Bilirubin", value=1.0)
    cholinesterase = st.number_input("Cholinesterase", value=7.0)
    cholesterol = st.number_input("Cholesterol", value=200.0)
    creatinina = st.number_input("Creatinine", value=1.0)
    gamma_glutamyl_transferase = st.number_input("Gamma GT", value=35.0)
    protein = st.number_input("Total Protein", value=7.0)

    sex = 1 if sex == "Male" else 0  # must match training encoding

    data = {
        "age": age,
        "sex": sex,
        "albumin": albumin,
        "alkaline_phosphatase": alkaline_phosphatase,
        "alanine_aminotransferase": alanine_aminotransferase,
        "aspartate_aminotransferase": aspartate_aminotransferase,
        "bilirubin": bilirubin,
        "cholinesterase": cholinesterase,
        "cholesterol": cholesterol,
        "creatinina": creatinina,
        "gamma_glutamyl_transferase": gamma_glutamyl_transferase,
        "protein": protein
    }

    return pd.DataFrame([data])

input_df = user_input_features()


num_cols = input_df.columns
input_scaled = scaler.transform(input_df[num_cols])


if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    predicted_label = le.inverse_transform(prediction)[0]
    confidence = np.max(prediction_proba) * 100

    st.success(f"🧾 **Predicted Condition:** {predicted_label}")
    st.info(f"📊 **Model Confidence:** {confidence:.2f}%")

    # Show probabilities: (using bargraphs)
    st.subheader("📈 Prediction Probabilities")
    prob_df = pd.DataFrame(
        prediction_proba,
        columns=le.classes_
    )
    st.bar_chart(prob_df.T)

st.markdown("---")
st.caption("⚕️ Educational purpose only. Not a medical diagnosis.")
