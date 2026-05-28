import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
import os
from sklearn.ensemble import RandomForestRegressor

model_path = 'model.pkl'
city_list_path = 'city_list.txt'
encoder_path = 'encoder.pkl'
scaler_path = 'scaler.pkl'

if not os.path.exists(model_path):
    df = pd.read_csv('realty_data.csv')
    df = df.drop(columns=['product_name', 'period', 'postcode', 'address_name', 'object_type', 'settlement',
                          'description', 'source', 'area', 'district', 'lat', 'lon', 'total_square'])
    df = df.dropna(subset=['city', 'rooms']).reset_index(drop=True)

    city_list = df['city'].unique().tolist()
    with open("city_list.txt", "w") as f:
        for item in city_list:
            f.write(f"{item}\n")

    encoder = OneHotEncoder(sparse_output=False)
    encoded_city = encoder.fit_transform(df['city'].to_numpy().reshape(-1, 1))
    encoded_city_df = pd.DataFrame(encoded_city, columns=encoder.get_feature_names_out())
    df_new = pd.concat([df, encoded_city_df], axis=1).drop(columns=['city'])

    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)

    X_train = df_new.drop(columns='price')
    y_train = df_new['price']
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    model = RandomForestRegressor(n_estimators=100, random_state=2026)
    model.fit(X_train_scaled, y_train)

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(city_list_path, "r") as f:
    city_list = [line.strip() for line in f]

st.header('Предсказание стоимости квартиры')
with st.form('flat_options'):
    city = st.selectbox("Город", city_list)
    rooms = st.slider('Количество комнат', 1, 15, 1)
    floor = st.slider('Этаж', 1, 60, 1)
    submitted = st.form_submit_button('Предсказать')


input_df = pd.DataFrame([{'rooms' : rooms, 'floor': floor}])
encoded_input_city = encoder.transform([[city]])
encoded_input_city_df = pd.DataFrame(encoded_input_city, columns=encoder.get_feature_names_out())
input_df = pd.concat([input_df, encoded_input_city_df], axis=1)
input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
predicted_price = round(model.predict(input_df_scaled)[0])
if submitted:
    st.write(f'Стоимость квартиры: {predicted_price:,} рублей'.replace(',', ' '))
