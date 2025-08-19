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

stock = st.selectbox("Select a Stock*: ", ('AAPL', 'TSLA', 'AMZN', 'MSFT', 'NVDA', 'GOOGL', 'META', 'NFLX', 'JPM', 'V', 'BAC', 'AMD', 'PYPL', 'DIS', 'T', 'PFE', 'COST', 'INTC', 'KO', 'TGT'))

#Data Summary: Based on Selected Stock
Stock_Data = Credit_Model_Data[Credit_Model_Data['symbol'] == stock]
Relevant_Years = Stock_Data['Year'].values

year = st.selectbox("Select a Year*: ", (Relevant_Years))

#%%Relevant Financial Data Summary

st.write("Financial Data ($)")
Stock_Year = Stock_Data[Stock_Data['Year'] == year] #Important for the rest


Stock_Year_Accounting_Data =  Stock_Year[['retainedEarnings', 'totalAssets', 
                                          'totalLiabilities','totalCurrentAssets', 'totalCurrentLiabilities', 
                                          'shortTermDebt','longTermDebt', 'workingCapital', 'revenue', 'ebit',
                                          'depreciationAndAmortization', 'netIncome', 
                                          'deferredIncomeTax', 'marketCapitalization', 'numberOfShares']]
Stock_Year_Accounting_Data.rename(columns = {'retainedEarnings': 'Retained Earnings', 'totalAssets': 'Total Assets', 
                                             'totalLiabilities': 'Total Liabilities', 'totalCurrentAssets': 'Total Current Assets',
                                             'totalCurrentLiabilities': 'Total Current Liabilities', 'shortTermDebt': 'Short Term Debt',
                                             'longTermDebt': 'Long Term Debt', 'workingCapital': 'Working Capital', 
                                             'revenue': 'Sales', 'ebit': 'EBIT', 
                                             'depreciationAndAmortization': 'Depreciation and Amortization', 
                                             'netIncome': 'Net Income', 'deferredIncomeTax': 'Deferred Income Tax', 
                                             'marketCapitalization': 'Market Capitalization', 'numberOfShares': 'Number of Shares'
                                             }, inplace = True)

#SYAD = Stock Year Accounting Data
SYAD_DF = Stock_Year_Accounting_Data.T.apply(lambda s: s.apply('{0:.2e}'.format))
SYAD_DF.rename(columns= {SYAD_DF.columns[0]:'Value ($)'}, inplace = True)

SYAD_DF_1 = SYAD_DF.iloc[:8]
SYAD_DF_2 = SYAD_DF.iloc[8:]

SYAD_col1, SYAD_col2 = st.columns(2)
with SYAD_col1:
    st.markdown(
    """
    <style>
    table td, table th {
        text-align: center !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.dataframe(SYAD_DF_1, use_container_width = True)
with SYAD_col2:
    st.markdown(
    """
    <style>
    table td, table th {
        text-align: center !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.dataframe(SYAD_DF_2, use_container_width = True)




st.write("Preview of Data")
