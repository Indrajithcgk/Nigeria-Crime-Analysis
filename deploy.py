import streamlit as st
import holidays
from datetime import datetime
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib

# Load the holidays for Nigeria
nigeria_holidays = holidays.Nigeria()

# Load the XGBoost model and ColumnTransformer
model_dir = 'artifacts/'
data_dir = 'data/'
model_XGB_SW = xgb.XGBClassifier()
model_XGB_SW.load_model(model_dir + 'model_XGB_SW.model')
column_transformer = joblib.load(model_dir + 'col_transformer.pkl')

# Load the transformed socio-demographic data
socio_demo_data = pd.read_csv(data_dir + 'socio_demo_for_model_prediction.csv')

def nigeria_day_info(date_string):
    date_obj = datetime.strptime(date_string, '%Y-%m-%d').date()
    day_of_month = date_obj.strftime('%d')
    month = date_obj.strftime('%m')

    is_holiday = 1 if date_obj in nigeria_holidays else 0
    is_weekday = 1 if date_obj.weekday() < 5 else 0

    return month, day_of_month, is_weekday, is_holiday

def predict_attack_prob(state, date_to_check):
    X_input = socio_demo_data.query("State == @state").copy()

    # Add the date related info
    X_input['month'], X_input['day'], X_input['isweekday'], X_input['is_holiday'] = nigeria_day_info(date_to_check)

    # Drop unnecessary columns
    X_input.drop('Unnamed: 0', axis=1, inplace=True)

    # Column transform input data
    X_input_transformed = column_transformer.transform(X_input)

    # Generate predictions
    pred_prob = np.round(model_XGB_SW.predict_proba(X_input_transformed)[0, 1] * 100, 1)

    return pred_prob

# Streamlit UI
st.title('Attack Probability Prediction')
st.write('Enter the State and Date to predict the probability of an attack.')

state = st.selectbox('Select State:', socio_demo_data['State'].unique())
date_to_check = st.date_input('Select Date:')
prediction_button = st.button('Predict Probability')

if prediction_button:
    prediction = predict_attack_prob(state, date_to_check.strftime('%Y-%m-%d'))
    st.write(f'Probability of an attack in {state} on {date_to_check.strftime("%Y-%m-%d")}: {prediction}%')