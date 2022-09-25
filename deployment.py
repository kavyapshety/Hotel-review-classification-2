# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 23:59:53 2022

@author: advai
"""

# Import libraries
import streamlit as st
import numpy as np
import pickle
import pandas as pd

# load the model from disk
loaded_model = pickle.load(open('E:/project file/trained_model.sav', 'rb'))



def Fraudulent_transaction(input_data):
    
    
    #Changing the input data to numpy array
    input_data_as_numpy_array = np.asarray (input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    #standardize the input data
    #scaler=StandardScaler()
    #std_data=scaler.fit_transform(input_data_reshaped)
    #print(std data)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
        return'The Transaction is not Fraudulent'
    else:
        return'The Transaction is Fraudulent'

def main():
    
    #giving a title
    st.title('Fraudulent Transaction Web App')
    
    
    
    #getting the input data from the user
   
    #Getting Input
    step = st.text_input('Time at wich Transaction happened 24hrs format')
    amount = st.text_input('Transaction amount')
    oldbalanceOrg = st.text_input('Old balance at Origin')
    newbalanceOrig = st.text_input('New balance at Origin')
    oldbalanceDest = st.text_input('Old balance at Destination')
    newbalanceDest = st.text_input('New balance at Destination')
    PAYMENT = st.text_input('if Transaction type PAYMENT then enter 1 or 0')
    DEBIT = st.text_input('if Transaction type DEBIT then enter 1 or 0')
    CASH_OUT = st.text_input('if Transaction type CASH_OUT then enter 1 or 0')
    TRANSFER = st.text_input('if Transaction type Transfer then enter 1 or 0')
    
    
    #code for prediction
    transaction=''

    
    #creating a button for prediction
    if st.button("model prediction result"):
        transaction= Fraudulent_transaction([step,PAYMENT,DEBIT,CASH_OUT,TRANSFER,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest])

    
    st.success(transaction)


if __name__=='__main__':
    main()



    