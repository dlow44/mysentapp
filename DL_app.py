import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Data preprocessing function
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

# Load dataset
s = pd.read_csv("social_media_usage.csv")
    
ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent": np.where(s["par"] == 1, 1, 0),
    "married": np.where(s["marital"] == 1, 1, 0),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"] > 98, np.nan, s["age"])
})

ss = ss.dropna()

# Define target variable (y) and features (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

# Train the logistic regression model
model = LogisticRegression(class_weight="balanced")
model.fit(X, y)

# Streamlit interface for user input
st.title("Linkedin Prediction")

st.write("Enter the attributes below to predict if the person is a Linkedin User:")

# Get user input for each feature
income = st.number_input("Income(1-9)", min_value=0, max_value=9, value=5)
education = st.number_input("Education Level (1-8)", min_value=1, max_value=8, value=4)
parent = st.selectbox("Are you a parent?", ["No", "Yes"])
married = st.selectbox("Are you married?", ["No", "Yes"])
female = st.selectbox("Are you female?", ["No", "Yes"])
age = st.number_input("Age", min_value=18, max_value=98, value=30)

# Convert categorical inputs to binary values
parent = 1 if parent == "Yes" else 0
married = 1 if married == "Yes" else 0
female = 1 if female == "Yes" else 0

# Prepare the input data for the model
user_data = pd.DataFrame({
    "income": [income],
    "education": [education],
    "parent": [parent],
    "married": [married],
    "female": [female],
    "age": [age]
})

# Function to make prediction based on user input
def make_prediction(user_data):
    # Use the trained model to make a prediction
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)

    return prediction, prediction_proba

# Make prediction when the button is clicked
if st.button("Predict"):
    prediction, prediction_proba = make_prediction(user_data)
    
    # Show the prediction and probability
    st.write(f"User Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
    st.write(f"Probability of Linkedin Use: {prediction_proba[0][1]:.2f} for Yes, {prediction_proba[0][0]:.2f} for No")

