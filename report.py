# New App

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return np.where(x == 1, 1, 0)



ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income":np.where(s["income"] > 9,np.nan, s['income']),
    "education":np.where(s["educ2"] >8, np.nan, s['educ2']),
    "parent":np.where(s["par"] == 1,1,0),
    "age":np.where(s["age"] > 98, np.nan, s['age']),
    "female":np.where(s["gender"] ==2,1,0),
     "married":np.where(s["marital"]== 1,1,0)})
  

ss = ss.dropna()


X = ss[['income', 'education', 'parent', 'age', 'female','married']]
y = ss['sm_li']

model = LogisticRegression(class_weight='balanced')
model.fit(X.values, y)

st.title('Linkedin User Prediction by Tyler Sovig')


education = st.selectbox("Education level",
    options = ["Less than high school",
               "High School Diploma",
               "High school graduate",
               "Some college, no degree",
               "Two-year associate degree",
               "Four-year college",
               "Some postgraduate or professional schooling, no postgraduate degree",
               "Postgraduate or professional schooling, including master's, doctorate, medical, or law degree"])

if education == "Less than high school" :education = 1
elif education == "High School Diploma": education = 2
elif education == "High school graduate": education = 3
elif education == "Some college, no degree" : education = 4
elif education == "Two-year associate degree" : education = 5
elif education == "Four-year college": education = 6
elif education == "Some postgraduate or professional schooling, no postgraduate degree": education = 7
else: education = 8


income= st.selectbox('Income Level', options=["Less than 10,000", 
                                      '10,000 - 20,000',
                                       '20,000 - 30,000',
                                       '30,000 - 40,000',
                                       '40,000 - 50,000',
                                       '50,000 - 75,000',
                                       '75,000 - 100,000',
                                       '100,000 - 150,000',
                                       '150,000 or more'])
if income ==" Less than 10,000" :income = 1
elif income == '10,000 - 20,000': income = 2
elif income == '20,000 - 30,000': income = 3
elif income == '30,000 - 40,000': income = 4
elif income == '40,000 - 50,000': income = 5
elif income == '50,000 - 75,000': income = 6
elif income == '75,000 - 100,000': income = 7
elif income == '100,000 - 150,000': income = 8 
else: income = 9

parent= st.radio('Are you a Parent?', options= [1,2])

age= st.slider( 'Age?', 1,97,1)

gender = st.radio('What is your Gender?', options= [1,2])

married= st.radio('Are you married?', options=[1,2])


user= [income, education, parent, age, gender, married]


st.write(f'The Odds of a User being on LinkedIn: {model.predict_proba([user])[0][1]*100}')

st.write(f'The Odds of a User being on LinkedIn: {model.predict([user])}')