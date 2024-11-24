import pandas as pd
import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

data = pd.read_csv('boston.csv', index_col=0)

features = data.drop(['PRICE'], axis=1)
average_vals = features.mean().values
property_stats = pd.DataFrame(data=average_vals.reshape(1, len(features.columns)), 
                              columns=features.columns)

st.title("House price prediction app")

st.divider()

st.write("This app uses advance machine learning to predict House valuation. You just have to fill a given set of input and You are ready to go")

st.divider()

st.write("**Enter property details to get an estimated price:**")

st.divider()

property_stats['RM'] = st.number_input("Number of rooms", min_value=0, value=0)
property_stats['PTRATIO'] = st.number_input("Number of student per classroom", min_value=0, value=0)
property_stats['DIS'] = st.number_input("Distance to town", min_value=0, value=0)
next_to_river = st.checkbox("Tick if the river passes near your House")
if next_to_river:
    property_stats['CHAS'] = 1
else:
    property_stats['CHAS'] = 0
property_stats['NOX'] = st.number_input("Pollution index", min_value=0.0, value=0.0, max_value=1.0, step=1e-6,
    format="%.2f")
property_stats['LSTAT'] = st.number_input("Percentage population below poverty", min_value=0.0, value=0.0, max_value=1.0, step=1e-6,
    format="%.2f")

st.divider()
predict_button = st.button("Predict")

if predict_button:

    st.balloons()
    log_prediction = model.predict(property_stats)[0]
    dollars = np.exp(log_prediction) * 1000
    st.write(f"The predicted price is {dollars:.6}")


else:
    st.write("please use predict button after entering value.")

