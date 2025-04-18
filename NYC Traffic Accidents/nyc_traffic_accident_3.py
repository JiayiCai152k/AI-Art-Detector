import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
st.set_page_config(
    page_title="NYC Traffic Accident",
    page_icon=":red_car:",
    layout="wide",
    initial_sidebar_state="expanded"
)

col1, col2 = st.columns([3, 1])
with col1:
    st.title("NYC Traffic Accident")
with col2:
    st.image("nypd-patch.png", width=100)

st.markdown("""
This application is a Streamlit dashboard that can be used to analyze traffic accidents in NYC 🗽💥🚗. 
The data includes information about accident location, date, time, contributing factors, and more. 
Use the sidebar radio button to navigate between different sections of the app.
""")

def load_data(): 
    df = pd.read_csv('NYC Accidents 2020.csv')
    return df
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
def draw_map_streamlit(df):
    st.subheader("Accident Locations")
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    st.map(df[["LATITUDE", "LONGITUDE"]])
    

def preprocess_data(df):
    st.header("Add preprocessing steps here")


def train_model():
    st.header("Add trainning steps here")

def predict():
     st.header("Add prediction steps here")


def show_images(image_path,caption):
    st.image(image_path, caption=caption, width=300)

def show_images_in_columns():
    col1, col2 = st.columns(2)
    with col1:
        image_path = 'GOHS logo transparent.png'
        show_images(image_path,'GOHS logo transparent')
    with col2:
        image_path = 'nypd-patch.png'

        show_images(image_path,'nypd-patch')

def main():
    df = load_data()

    show_images_in_columns()
    tabs = ["Show Dataset", "Show Features", "Describe Features", "Show Map", "Preprocessing", "Training", "Prediction"]
    
    selected_tab = st.sidebar.radio("Choose a functionality", tabs)

    if selected_tab == "Show Dataset":
        explore_data(df, "NYC Traffic Accident Dataset")
    elif selected_tab == "Show Features":
        st.write(df.columns)
    elif selected_tab == "Describe Features":
        describe_features(df)
    elif selected_tab == "Show Map":
        draw_map_streamlit(df)
    elif selected_tab == "Preprocessing":
        preprocess_data(df)
    elif selected_tab == "Training":
        train_model()
    elif selected_tab == "Prediction":
        predict()
    

if __name__ == '__main__':
    main()
