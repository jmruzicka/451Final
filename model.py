import numpy as np 
import pandas as pd 
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image

# Load  model a 
model = joblib.load(open("./model1.joblib","rb")) #"C:\CS-451\FinalProject\451Final\model1.joblib"

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
    #data.Postseason = data.Postseason.map({1:'Champions', 2:'Runner-Up', 4:'Final Four', 8:'Elite Eight', 16:'Sweet Sixteen', 32:'Round of 32', 64: 'Round of 64', 128:'No Dancing for your team'})


    data = (prediction_proba[0]*100).round(2)
    #data.POSTSEASON = data.POSTSEASON.map({1:'Champions', 2:'Runner-Up', 4:'Final Four', 8:'Elite Eight', 16:'Sweet Sixteen', 32:'Round of 32', 64: 'Round of 64', 128:'No Dancing for your team'})
    grad_percentage = pd.DataFrame(data = data,columns = ['POSTSEASON'],index = ['Champs','Champs', 'Runners-Up', 'Final Four', 'Elite Eight', 'Sweet Sixteen', 'Round of 32', 'Round of 64', 'No Dancing for your team'])
    #grad_percentage = pd.DataFrame(data = data,columns = ['POSTSEASON'],index = ['Champions','Runner-Up','Final Four','Elite Eight','Sweet Sixteen','Round of 32','Round of 64', 'No Dancing for your team'])
    #grad_percentage.rename({0:"Champions",1: 'Champions', 2: 'Runner-Up', 4: 'Final Four', 8: 'Elite Eight', 16: 'Sweet Sixteen', 32: 'Round of 32', 64: 'Round of 64', 128: 'No Dancing for your team'})
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
    ax.set_ylabel("Tournament Finish", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return

st.write("""
# March Madness Finish Prediction 
This app predicts the postseason finish using basketball statistics input via the **side panel** 
""")

#read in wine image and render with streamlit
image = Image.open('./March_Madness_logo.svg.png')
st.image(image, caption='March Madness',use_column_width=True)
#image2 = Image.open('C:/CS-451/FinalProject/451Final/key.png')
#st.image(image2, caption='Graph Key',use_column_width=True)

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe
    """
  
    ADJOE = st.sidebar.slider('Adjusted Offensive Efficiency', 80.0, 130.0, 120.0, 0.1)
    ADJDE = st.sidebar.slider('Adjusted Defensive Efficiency', 80.0, 130.0, 84.0, 0.1)
    BARTHAG  = st.sidebar.slider('Power Ranking', 0.0, 1.0, 0.8, 0.001)
    #WAB  = st.sidebar.slider('Wins Above Bubble', -25.0, 15.0, 7.5, 0.01)
    EFG_O  = st.sidebar.slider('Effective Field Goal Percentage', 39.0, 60.0, 52.0,0.1)
    EFG_D = st.sidebar.slider('Effective Field Goal Allowed', 35.0, 60.0, 47.0,0.1)
    FTRD = st.sidebar.slider('Free Throws Allowed', 20.0, 60.0, 30.0,0.1)
    WAB  = st.sidebar.slider('Wins Above Bubble', -25.0, 15.0, 7.5, 0.01)
  
    
    features = {
            'ADJOE': ADJOE,
            'ADJDE': ADJDE,
            'BARTHAG': BARTHAG,
            'EFG_O': EFG_O,
            'EFG_D': EFG_D,
            'FTRD': FTRD,
            'WAB': WAB
            }
    data = pd.DataFrame(features,index=[0])
    #data = data.tail(data.shape[0] -1)
    #data.drop(columns=data.columns[0], axis=1, inplace=True)
    #data.drop(index=data.index[0],  axis=0,  inplace=True)
    #data = data.iloc[1:]
    #print(data.loc[0])
    #df1 = df.iloc[1:]
    return data

user_input_df = get_user_input()
#print('test')
#print(user_input_df)
processed_user_input = data_preprocessor(user_input_df)
#print(processed_user_input)
st.subheader('User Input parameters')
st.write(user_input_df)
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_option('browser.gatherUsageStats', False)

prediction = model.predict(processed_user_input)
#print(prediction)
#st.write(prediction)
prediction_proba = model.predict_proba(processed_user_input)

visualize_confidence_level(prediction_proba)
