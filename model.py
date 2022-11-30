import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image


# Load  model a 
model = joblib.load(open("c:/CS-451/FinalProject/451Final/model1.joblib","rb")) #"C:\CS-451\FinalProject\451Final\model1.joblib"

def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    #df.Postseason = df.Postseason.map({1:'Champions', 2:'Runner-Up', 4:'Final Four', 8:'Elite Eight', 16:'Sweet Sixteen', 32:'Round of 32', 64: 'Round of 64', 128:'No Dancing for your team'})
    return df

def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['POSTSEASON'],index = ['Champions','Runner-Up','Final Four','Elite Eight','Sweet Sixteen','Round of 32','Round of 64', 'No Dancing for your team'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel(" Percentage(%) Confidence Level", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Finish", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return

st.write("""
# March Madness Finish Prediction 
This app predicts the postseason finish using basketball statistics input via the **side panel** 
""")

#read in wine image and render with streamlit
image = Image.open('C:/CS-451/FinalProject/451Final/March_Madness_logo.svg.png')
st.image(image, caption='March Madness',use_column_width=True)

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe
    """
  # wine_type = st.sidebar.selectbox("Select Wine type",("white", "red"))
    ADJOE = st.sidebar.slider('Adjusted Offensive Efficiency', 80, 130, 110)
    ADJDE = st.sidebar.slider('Adjusted Defensive Efficiency', 80, 130, 90)
    BARTHAG  = st.sidebar.slider('Power Ranking', 0.0, 1.0, 0.6)
    WAB  = st.sidebar.slider('Wins Above Bubble', -25, 15, 0)
    EFG  = st.sidebar.slider('Effective Field Goal Percentage', 39, 60, 52)
    EFG_D = st.sidebar.slider('Effective Field Goal Allowed', 35, 60, 47)
    FTRD = st.sidebar.slider('Free Throws Allowed', 20, 60, 30)
    #density = st.sidebar.slider('density', 0.98, 1.03, 1.0)
   # pH = st.sidebar.slider('pH', 2.72, 4.01, 3.0)
    #sulphates = st.sidebar.slider('sulphates', 0.22, 2.0, 1.0)
   # alcohol = st.sidebar.slider('alcohol', 8.0, 14.9, 13.4)
    
    features = {
            'ADJOE': ADJOE,
            'ADJDE': ADJDE,
            'BARTHAG': BARTHAG,
            'WAB': WAB,
            'EFG': EFG,
            'EFG_D': EFG_D,
            'FTRD': FTRD
            
            }
    data = pd.DataFrame(features,index=[0])

    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)

st.subheader('User Input parameters')
st.write(user_input_df)

prediction = model.predict(processed_user_input)
prediction_proba = model.predict_proba(processed_user_input)

visualize_confidence_level(prediction_proba)
