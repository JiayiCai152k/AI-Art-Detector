import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



st.set_page_config(
    page_title="NYC Traffic Accident",
    page_icon=":red_car:",
    layout="wide",
    initial_sidebar_state="expanded",
    
)

col1, col2 = st.columns([3, 1])
with col1:
    st.title("NYC Traffic Accident")
with col2:
    st.image("nypd-patch.png", width=100)

st.markdown("""
This application is a Streamlit dashboard that can be used to analyze traffic accidents in NYC 🗽💥🚗. 
The data includes information about accident location, date, time, contributing factors, and more. 
Use the sidebar menu to navigate between different sections of the app.
""")
def load_data():
    df = pd.read_csv('NYC Accidents 2020.csv')
    return df

def show_raw_data(df):
    st.subheader('Raw Data')
    st.write(df)
def explore_data(df, subheader):
    st.subheader(subheader)
    
    st.write(df)

def describe_features(df):
    st.subheader('Features Description')
    features_description = pd.DataFrame({
        'Feature': df.columns,
        'Description': [
            'Crash date',
            'Crash time',
            'Borough',
            'ZIP Code',
            'Latitude',
            'Longitude',
            'Location',
            'On street name',
            'Cross street name',
            'Off street name',
            'Number of persons injured',
            'Number of persons killed',
            'Number of pedestrians injured',
            'Number of pedestrians killed',
            'Number of cyclist injured',
            'Number of cyclist killed',
            'Number of motorist injured',
            'Number of motorist killed',
            'Contributing factor vehicle 1',
            'Contributing factor vehicle 2',
            'Contributing factor vehicle 3',
            'Contributing factor vehicle 4',
            'Contributing factor vehicle 5',
            'Collision ID',
            'Vehicle type code 1',
            'Vehicle type code 2',
            'Vehicle type code 3',
            'Vehicle type code 4',
            'Vehicle type code 5'
        ]
    })
    st.write(features_description)

def draw_map(df):
    st.subheader("Accident Locations")
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    st.map(df[["LATITUDE", "LONGITUDE"]])
def preprocess_data(df):
    st.header("Add some preprocessing steps here")


def train_model():
    st.header("Add trainning steps here")

def predict():
     st.header("Add prediction steps here")
def main():
    df = load_data()

    # Create a selectboxed menu at sidebar
    menu = ['Data Exploration', 'Show Features', 'Feature Description', "Show Map", "Preprocessing", "Training", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Data Exploration':
        explore_data(df,subheader='Data Exploration')
    elif choice == 'Show Features':
        explore_data(df.columns, 'Show Features')
    elif choice == 'Feature Description':
        describe_features(df)
    elif choice == 'Show Map':
        draw_map(df)
    elif choice == 'Preprocessing':
        preprocess_data(df)
    elif choice == 'Training':
        train_model()
    elif choice == 'Prediction':
        predict()

        

if __name__ == '__main__':
    main()