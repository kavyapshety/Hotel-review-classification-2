# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 11:16:21 2022

@author: advai
"""
# Importing the libraries
import streamlit as st
import pickle
import spacy
import numpy as np
import pandas as pd
import random
from PIL import Image
import joblib
import streamlit.components.v1 as components
from joblib import dump, load
import joblib
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

#load model
# load the model from disk
with open('C:/Users/advai/hotel project/Hotel Reviews final.pkl','rb') as f:
        model=joblib.load(f)

# Body of the application
st.header("Hotel Review Prediction Application.")
st.markdown("This application is trained on machine learning model.\n "
            "This application can predict if the given **review**"
            " is **Positive, Negative or Neutral**")


text = st.text_input("Type your review here...", """""")



# Preprocessing the text
nlp = spacy.load("en_core_web_lg")

def preprocessing(text):
    """Takes the text input and removes the stop words and punctuations from the text and gives processed text output.
    """
    global nlp
    doc = nlp(text)

    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)


preprocessed_input = [preprocessing(text)]
st.write(preprocessed_input)

# Tfidf


# Making prediction
if st.button("Click to make prediction"):

    # Making prediction for model input
    prediction = int(loaded_model.predict(preprocessed_input))
    st.write(prediction)

    # Returning true prediction
    if prediction == -1:
        st.write("Negative")
    elif prediction == 1:
        st.write("Positive")
    else:
        st.write("Neutral")
else:
    pass

def main():
    # set page title
    st.set_page_config('NLP classification')

    st.title('Hotel review classification')
    # Using "with" notation
    with st.sidebar:
        image= Image.open("iidt_logo_138.jpeg")
        add_image=st.image(image,use_column_width=True)

    social_acc = ['About']
    social_acc_nav = st.sidebar.selectbox('About', social_acc)
    if social_acc_nav == 'About':
        st.sidebar.markdown("<h2 style='text-align: center;'> This Project completed under ExcelR, the team completed the project:</h2> ", unsafe_allow_html=True)
        st.sidebar.markdown('''---''')
        st.sidebar.markdown('''
        • Miss. Kavya M P \n
        • Mr. Raju A S \n
        • Mr. AMAN SAJAD \n 
        • Mr.Raghavendra R \n
        ''')
        st.sidebar.markdown("[ Visit To Github Repositories](https://github.com/kavyapshety/Hotel-review-classification-2.git)")   
    menu_list = [""]
    menu = st.radio("Menu", menu_list)
               
    if menu == 'Hotel-review-classification-2':
            
            st.title("Hotel-review-classification-2")
            #import the image
            image= Image.open("header_hotel_rating_classification.jpeg")
            st.image(image,use_column_width=True)

            html_temp = """
            <div style="background-color:tomato;padding:10px">
            <h2 style="color:white;text-align:center;">Hotel review classification </h2>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)