import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew

st.write("""
# Simple Car Price Prediction Tool

The online app allows you to predict Car price according to 
various independet features, such as:

Model, Type, Age, Km etc. 
Use the slider on  the lef-hand side to set your features and calculate 
price of the car.

The dataset was taken  from "https://www.kaggle.com/hellbuoy/car-price-prediction/data"


""")

df = pd.read_csv('CarPrice_data.csv')


target = "Car price"

encode = "horsepower, stroke"

st.sidebar.header('Car Input Parameters')

def user_input_features():
	engine_size = st.sidebar.slider("min. enginesize", 40, 200, 40)
	fuel_type = st.sidebar.selectbox("Which fuel type", df["fueltype"].unique())
	horse_power = st.sidebar.slider("min. horsepower", 50, 200, 40)
	data = {'engine_size' : engine_size, 'fuel_type': fuel_type, 'horse_power': horse_power}
	features = pd.DataFrame(data, index=[0])
	return features

def reg_anal():
	#X_in = df2.horse_power[0]
	features = user_input_features()
	X_in = features.horse_power[0]
	Y_pred = -3721.7614943227563 + 163.263*X_in
	return (Y_pred , X_in, features)

regvals = reg_anal()


st.subheader('User input parameters')

st.write(regvals[2])

st.subheader ("""start predicting""")
st.write("Using dataset from https://www.kaggle.com/hellbuoy/car-price-prediction/data")

# define x and y variables
price_data = df[["horsepower", "price"]]

x = price_data["horsepower"]
y = price_data["price"]
	
st.bar_chart(price_data)


st.subheader('Predicted Price using linear Regreassion Model')




st.write("The predicted price for your car is:", regvals[0], "$")

#st.write(regvals[2])


st.subheader("Raw data")
df