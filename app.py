# app.py
import streamlit as st
import pandas as pd
import torch
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from uda_regression_sab import DANNRegressor, predict_rows, load_preprocessor, MODEL_PATH

st.set_page_config(page_title="Heart Rate Prediction Dashboard", page_icon="❤️", layout="wide")

# Styling/theme toggle
theme = st.sidebar.radio("Theme", ("Light", "Dark"), index=0)
if theme == "Light":
    st.markdown("""
        <style>
        body {background-color: white; color: black;}
        .stButton>button {background-color: #0066ff; color: white;}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {background-color: #121212; color: white;}
        .stButton>button {background-color: #00bfa5; color: black;}
        </style>
    """, unsafe_allow_html=True)

st.sidebar.title("💓 Heart Rate Dashboard")
st.sidebar.markdown("Predict `heart_rate_apache` using a DANN model.")

# Load preprocessor (fitted during training)
preprocessor = load_preprocessor()

# Build a dummy row to compute input_dim (handles numeric + one-hot expansion)
# You must supply the same numeric & categorical column lists used in training:
numeric_cols = [
    'age','bmi','weight','height','d1_heartrate_max','d1_heartrate_min',
    'd1_mbp_max','d1_mbp_min','d1_sysbp_max','d1_sysbp_min','d1_diasbp_max',
    'd1_diasbp_min','d1_resprate_max','d1_resprate_min','d1_spo2_max','d1_spo2_min',
    'd1_bun_max','d1_bun_min','d1_creatinine_max','d1_creatinine_min','d1_sodium_max',
    'd1_sodium_min','d1_potassium_max','d1_potassium_min','d1_glucose_max','d1_glucose_min',
    'd1_albumin_max','d1_albumin_min','map_apache','aids','cirrhosis','hepatic_failure',
    'leukemia','lymphoma','solid_tumor_with_metastasis'
]
categorical_cols = ['gender', 'ethnicity']

# Create a safe dummy with numeric zeros and common categories to compute input dims
dummy = {c: 0.0 for c in numeric_cols}
# choose safe values for categorical - must be among categories seen in training,
# but OneHotEncoder(handle_unknown='ignore') was used during training so unknown categories are fine.
dummy['gender'] = 'M'
dummy['ethnicity'] = 'Caucasian'
input_dim = preprocessor.transform(pd.DataFrame([dummy])).shape[1]

# Instantiate and load model with computed input_dim
model = DANNRegressor(input_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Layout - tabs
tab1, tab2, tab3 = st.tabs(["📝 Single Prediction", "📂 Batch Prediction", "📊 Visualization"])

with tab1:
    st.header("Single Patient Prediction")
    input_data = {}
    cols = st.columns(3)
    for i, col in enumerate(numeric_cols):
        input_data[col] = cols[i % 3].number_input(col, value=0.0)
    for col in categorical_cols:
        options = ['M','F'] if col == 'gender' else ['Caucasian','Hispanic','Other']
        input_data[col] = st.selectbox(col, options=options)
    if st.button("Predict Single Patient"):
        df = pd.DataFrame([input_data])
        preds = predict_rows(df, model, preprocessor)
        st.success(f"Predicted heart_rate_apache: {preds[0]:.2f}")
        st.balloons()

with tab2:
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv','xlsx'])
    if uploaded_file:
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("Preview (first rows):")
        st.dataframe(df.head())
        if st.button("Predict Batch"):
            preds = predict_rows(df, model, preprocessor)
            df['predicted_heart_rate_apache'] = preds
            st.dataframe(df.head())
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

with tab3:
    st.header("Visualization")
    viz_file = st.file_uploader("Upload CSV with actual & predicted columns (for metrics/plots)", type=['csv','xlsx'])
    if viz_file:
        if viz_file.name.lower().endswith('.csv'):
            viz_df = pd.read_csv(viz_file)
        else:
            viz_df = pd.read_excel(viz_file)
        st.dataframe(viz_df.head())
        required = ['heart_rate_apache', 'predicted_heart_rate_apache']
        if all(c in viz_df.columns for c in required):
            mae = mean_absolute_error(viz_df['heart_rate_apache'], viz_df['predicted_heart_rate_apache'])
            mse = mean_squared_error(viz_df['heart_rate_apache'], viz_df['predicted_heart_rate_apache'])
            rmse = mean_squared_error(viz_df['heart_rate_apache'], viz_df['predicted_heart_rate_apache'], squared=False)
            r2 = r2_score(viz_df['heart_rate_apache'], viz_df['predicted_heart_rate_apache'])
            cols = st.columns(4)
            cols[0].metric("MAE", f"{mae:.2f}")
            cols[1].metric("MSE", f"{mse:.2f}")
            cols[2].metric("RMSE", f"{rmse:.2f}")
            cols[3].metric("R²", f"{r2:.2f}")

            viz_df['error'] = viz_df['predicted_heart_rate_apache'] - viz_df['heart_rate_apache']
            fig = px.scatter(viz_df, x='heart_rate_apache', y='predicted_heart_rate_apache', color='error',
                             color_continuous_scale='RdYlGn_r', title='Predicted vs Actual')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Uploaded file must contain columns: {required}")
