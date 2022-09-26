import nltk
nltk.download('wordnet')

import warnings
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
import re
from rake_nltk import Rake
import pickle
import streamlit as st
import numpy as np
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# Warnings ignore 
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


# loading the trained model
filename = open('svc_model_fitted.pkl', 'rb') 
model = pickle.load(filename)

def main():
    # set page title
    st.set_page_config('Hotel review')

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
        • Miss. Payal vishal kathar \n
        • Miss. Priyanka sunil mahule \n 
        • Mr.Nidhin K V \n
        ''')
        st.sidebar.markdown("[ Visit To Github Repositories](https://github.com/kavyapshety/Hotel-review-classification-2.git)")   
    menu_list = ["Hotel-review-classification-2"]
    menu = st.radio("Menu", menu_list)

 if menu == 'Hotel-review-classification-2':
            
            st.title("Hotel-review-classification-2")
            #import the image
            image= Image.open("Header_hotel_rating_classification.jpeg")
            st.image(image,use_column_width=True)

            html_temp = """
            <div style="background-color:tomato;padding:10px">
            <h2 style="color:white;text-align:center;">hotel review classification </h2>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

# Title of the application 
st.title('Text analysis \n', )
st.header("Sentiment Analysis")
st.subheader("Enter the text here")

input_text = st.text_area("Enter the review", height=50)

# Sidebar options
option = st.sidebar.selectbox('Navigation',['Sentiment Analysis','Word Cloud'])
st.set_option('deprecation.showfileUploaderEncoding', False)
if option == "Sentiment Analysis":
    
    
    
    if st.button("Predict sentiment"):
        st.write("Number of words in Review:", len(input_text.split()))
        wordnet=WordNetLemmatizer()
        text=re.sub('[^A-za-z0-9]',' ',input_text)
        text=text.lower()
        text=text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        filename = open('svc_model_fitted.pkl', 'rb') 
        model = pickle.load(filename)
        
        
        if model.predict(transformed_input) ==0:
            st.write("Input review has Negative Sentiment.:sad:")
        elif model.predict(transformed_input) ==2:
            st.write("Input review has Positive Sentiment.:smile:")
        else:
            st.write(" Input review has Neutral Sentiment.😐")
         

elif option == "Word Cloud":
    st.header("Word cloud")
    if st.button("Generate Wordcloud"):
        wordnet=WordNetLemmatizer()
        text=re.sub('[^A-za-z0-9]',' ',input_text)
        text=text.lower()
        text=text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        wordcloud = WordCloud().generate(text)
        plt.figure(figsize=(40, 30))
        plt.imshow(wordcloud) 
        plt.axis("off")
        
        st.pyplot()
