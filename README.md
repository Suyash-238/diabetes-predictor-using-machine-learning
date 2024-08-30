# diabetes-predictor-using-machine-learning
here is the implementation of python


#pip install pandas
#pip install sklearn


# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv(r'C:\Users\Admin\Desktop\PythonProject\DiabetesDetection\diabetes.csv')

st.title('Diabetes Checkup')

st.subheader('Training Data')
st.write(df.describe())
df = pd.read_csv(r'C:\Users\Admin\Desktop\DiabetesDetection\diabetes.csv')

st.subheader('Visualisation')
st.bar_chart(df)
# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())


# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# FUNCTION
def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 120 )
@@ -35,7 +43,7 @@ def user_report():
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.slider('Age', 21,88, 33 )

  user_report = {
  user_report_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
@@ -45,27 +53,46 @@ def user_report():
      'dpf':dpf,
      'age':age
  }
  report_data = pd.DataFrame(user_report, index=[0])
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)




# MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)

st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')

user_result = rf.predict(user_data)
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are healthy'
else:
  output = 'You are not healthy'
st.title(output)




st.write(output)

# VISUALISATIONS
fig_pregnancies = plt.figure(figsize = (10,6))
ax1 = sns.distplot(df['Pregnancies'] , bins = 100)
ax2 = sns.distplot(user_data['pregnancies'])
st.pyplot(fig_pregnancies)

fig_glucose = plt.figure()
ax1 = sns.scatterplot(x = 'Outcome', y = 'Glucose', data = df , hue = 'Outcome')
ax2 = sns.scatterplot(y = user_data['glucose'], x = user_result[0], color='red', s = 100)
st.pyplot(fig_glucose)
