# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 12:40:43 2025

@author: uaniekwe1
"""

import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
import numpy as np

# Load model and top 10 features
ridge_top10_model = joblib.load('ridge_top10_model.pkl')
top_10_features = joblib.load('top10_features.pkl')

# Page config
st.set_page_config(page_title="Saving Predictor", page_icon="💰")

# Custom title
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🔮 Efficiency Filter Saving Predictor</h1>", 
    unsafe_allow_html=True
)

st.write("Upload your Excel file to predict the Saving values based on your data.")

# Upload file
uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    sheet_name = st.text_input("Enter sheet name (leave blank for first sheet):", value='')

    try:
        with st.spinner('Reading your file and making predictions... ⏳'):
            # Read Excel
            if sheet_name.strip() == '':
                df_new = pd.read_excel(uploaded_file)
            else:
                df_new = pd.read_excel(uploaded_file, sheet_name=sheet_name)

            # Check features
            missing_features = [feat for feat in top_10_features if feat not in df_new.columns]
            if missing_features:
                st.error(f"Missing features in your file: {missing_features}")
            else:
                X_new = df_new[top_10_features]
                y_pred_new = ridge_top10_model.predict(X_new)

                # Calculate Confidence Interval (assuming residual variance)
                # Here we simulate a residual standard deviation (sigma)
                if hasattr(ridge_top10_model, 'rmse_'):
                    rmse = ridge_top10_model.sigma_
                else:
                    rmse = 0.6  # From trained model

                z_score = 1.96  # for 95% CI 2xstd error, 1 for 68% CI=> 1 std error
                error_margin = z_score * rmse

                lower_bounds = y_pred_new - error_margin
                upper_bounds = y_pred_new + error_margin

                result_df = pd.DataFrame({
                    'Predicted_Saving': y_pred_new,
                    'Lower_Confidence Interval': lower_bounds,
                    'Upper_Confidence Interval': upper_bounds
                })

                st.success('✅ Prediction Complete!')
                st.dataframe(result_df.style.background_gradient(cmap="YlGn"), use_container_width=True)

                # Prepare download
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False)
                excel_data = output.getvalue()

                st.download_button(
                    label="Download Predicted Results (Excel)",
                    data=excel_data,
                    file_name="Predicted_Saving_with_CI.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👆 Upload a file to get started!")

# Footer
st.markdown(
    "<hr style='margin-top: 50px;'>"
    "<div style='text-align: center;'>Made with ❤️ by Uchenna</div>", 
    unsafe_allow_html=True
)
