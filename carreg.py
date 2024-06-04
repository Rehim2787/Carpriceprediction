import streamlit as st
import pickle
import numpy as np
import json


with open('car_random_forest.pkl', 'rb') as file:
    model = pickle.load(file)

with open('robust_scaler.pkl', 'rb') as file:
    robust_scaler = pickle.load(file)

with open('model_mapping.json', 'r') as file:
    model_mapping = json.load(file)

with open('brand_mapping.json', 'r') as file:
    brand_mapping = json.load(file)

with open('type_of_ban_mapping.json', 'r') as file:
    type_of_ban_mapping = json.load(file)

st.title('Car Price Prediction App')

views = st.number_input('Number of Views', min_value=0, max_value=1000000, value=500)
distance = st.number_input('Distance Driven (in km)', min_value=0, max_value=1000000, value=50000)
release_date = st.number_input('Release Year', min_value=1900, max_value=2024, value=2015)
horsepower = st.number_input('Horsepower', min_value=0, max_value=2000, value=150)

model_selected = st.selectbox('Model', list(model_mapping.keys()))
brand_selected = st.selectbox('Brand', list(brand_mapping.keys()))
type_of_ban_selected = st.selectbox('Type of ban', list(type_of_ban_mapping.keys()))

model_kfold_target_enc = model_mapping[model_selected]
brand_kfold_target_enc = brand_mapping[brand_selected]
type_of_ban_kfold_target_enc = type_of_ban_mapping[type_of_ban_selected]


if st.button('Predict Price'):
    features = np.array([[release_date, distance, views, horsepower, model_kfold_target_enc, brand_kfold_target_enc,type_of_ban_kfold_target_enc]])
    features = robust_scaler.transform(features)
    
    log_predicted_price = model.predict(features)
    predicted_price = np.exp(log_predicted_price)

    st.write(f'The predicted price of the car is {predicted_price[0]:,.2f} AZN')
if __name__ == "__main__":
    main()
