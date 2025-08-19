# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 21:24:01 2025

@author: ManishArora
"""

import streamlit as st
import pandas as pd

st.title('Credit Risk Models')
st.write('Author: Manish Rajkumar Arora')
st.write("DataSources: FRED, YahooFinance, FMP.")

url_CreditModel = "https://raw.githubusercontent.com/man-aro/Credit-Risk-Model/main/Dataset_Relevant_CreditModel.csv"
url_KMVData = "https://raw.githubusercontent.com/man-aro/Credit-Risk-Model/tree/main/KMV_Merton/"

Credit_Model_Data = pd.read_csv(url_CreditModel)
Credit_Model_Data.columns
stock = st.selectbox("Select a Stock*: ", ('AAPL', 'TSLA', 'AMZN', 'MSFT', 'NVDA', 'GOOGL', 'META', 'NFLX', 'JPM', 'V', 'BAC', 'AMD', 'PYPL', 'DIS', 'T', 'PFE', 'COST', 'INTC', 'KO', 'TGT'))

#Data Summary: Based on Selected Stock
Credit_Score_Data = Credit_Model_Data[Credit_Model_Data['symbol'] == stock]
Relevant_Years = Credit_Score_Data['Year'].values

stock = st.selectbox("Select a Stock*: ", (Relevant_Years))





st.write("Preview of Data")