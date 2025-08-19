# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 21:24:01 2025

@author: ManishArora
"""

import streamlit as st
import pandas as pd
import numpy as np

st.title('Credit Risk Models')
st.write('Author: Manish Rajkumar Arora')
st.write("DataSources: FRED, YahooFinance, FMP.")

url_CreditModel = "https://raw.githubusercontent.com/man-aro/Credit-Risk-Model/main/Dataset_Relevant_CreditModel.csv"
url_KMVData = "https://raw.githubusercontent.com/man-aro/Credit-Risk-Model/tree/main/KMV_Merton/"

Credit_Model_Data = pd.read_csv(url_CreditModel)

stock = st.selectbox("Select a Stock: ", ('AAPL', 'TSLA', 'AMZN', 'MSFT', 'NVDA', 'GOOGL', 'META', 'NFLX', 'JPM', 'V', 'BAC', 'AMD', 'PYPL', 'DIS', 'T', 'PFE', 'COST', 'INTC', 'KO', 'TGT'))

#Data Summary: Based on Selected Stock
Stock_Data = Credit_Model_Data[Credit_Model_Data['symbol'] == stock]
Relevant_Years = Stock_Data['Year'].values

year = st.selectbox("Select a Year: ", (Relevant_Years))

#%%Relevant Financial Data Summary
st.write("Financial Data Summary")
Stock_Year = Stock_Data[Stock_Data['Year'] == year] #Important for the rest


Stock_Year_Accounting_Data =  Stock_Year[['retainedEarnings', 'totalAssets', 
                                          'totalLiabilities','totalCurrentAssets', 'totalCurrentLiabilities', 
                                          'shortTermDebt','longTermDebt', 'workingCapital', 'revenue', 'ebit',
                                          'depreciationAndAmortization', 'netIncome', 
                                          'deferredIncomeTax', 'marketCapitalization', 'numberOfShares', 'GNP_PriceIndexLevel']]
Stock_Year_Accounting_Data.rename(columns = {'retainedEarnings': 'Retained Earnings', 'totalAssets': 'Total Assets', 
                                             'totalLiabilities': 'Total Liabilities', 'totalCurrentAssets': 'Total Current Assets',
                                             'totalCurrentLiabilities': 'Total Current Liabilities', 'shortTermDebt': 'Short Term Debt',
                                             'longTermDebt': 'Long Term Debt', 'workingCapital': 'Working Capital', 
                                             'revenue': 'Sales', 'ebit': 'EBIT', 
                                             'depreciationAndAmortization': 'Depreciation and Amortization', 
                                             'netIncome': 'Net Income', 'deferredIncomeTax': 'Deferred Income Tax', 
                                             'marketCapitalization': 'Market Capitalization', 'numberOfShares': 'Number of Shares',
                                             'GNP_PriceIndexLevel':'GNP (Index = 1968)'
                                             }, inplace = True)

#SYAD = Stock Year Accounting Data
SYAD_DF = Stock_Year_Accounting_Data.T.apply(lambda s: s.apply('{0:.2e}'.format))
SYAD_DF.rename(columns= {SYAD_DF.columns[0]:'Value ($)'}, inplace = True)

SYAD_DF_1 = SYAD_DF.iloc[:8]
SYAD_DF_2 = SYAD_DF.iloc[8:]

SYAD_col1, SYAD_col2 = st.columns(2)
with SYAD_col1:
    st.dataframe(SYAD_DF_1, use_container_width = True)
with SYAD_col2:
    st.dataframe(SYAD_DF_2, use_container_width = True)



#%%Credit Score Data and Computation

st.write("Select Credit Score")

score = st.selectbox("Select Credit Score: ", ('Altman Z-Score', 'Ohlson O-Score'))

Score_DF = Stock_Year[['Altman_A', 'Altman_B', 'Altman_C', 'Altman_D', 'Altman_E', 
                       'Ohlson_A', 'Ohlson_B', 'Ohlson_C', 'Ohlson_D', 'Ohlson_E', 'Ohlson_F', 
                       'Ohlson_G', 'Ohlson_H', 'Ohlson_I']]

def AltmanZScore(A, B, C, D, E):
    Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
    return Z 

def OhlsonOScore(A, B, C, D, E, F, G, H, I):
    O_Score = -1.32 -0.407*np.log(A) + 6.03*B - 1.43*C + 0.0757*D - 1.72*E - 2.37*F -1.83*G + 0.285*H - 0.521*I
    return O_Score

def Prob(X):
    Prob = np.exp(X)/(1 + np.exp(X))
    return Prob * 100

if score == 'Altman Z-Score':
    Altman_DF = Stock_Year[['Altman_A', 'Altman_B', 'Altman_C', 'Altman_D',
                            'Altman_E']]
    Altman_DF['Z-Score'] = Altman_DF.apply(lambda row: AltmanZScore(row['Altman_A'], row['Altman_B'], 
                                                                 row['Altman_C'], row['Altman_D'], 
                                                                 row['Altman_E']), axis = 1)
    Altman_DF.rename(columns = {'Altman_A': 'WC/TA', 'Altman_B':'RE/TA', 'Altman_C':'EBIT/TA', 
                                'Altman_D': 'MktCap/TL', 'Altman_E':'Sales/TA'}, inplace = True)
    
    Altman_DF = Altman_DF.apply(lambda s: s.apply('{0:.2f}'.format))
    Altman_DF_T = Altman_DF.T
    Altman_DF_T.rename(columns= {Altman_DF_T.columns[0]:' '}, inplace = True)
    Altman_DF_T1 = Altman_DF_T.iloc[:3]
    Altman_DF_T2 = Altman_DF_T.iloc[3:]
    
    Altman_col1, Altman_col2 = st.columns(2)
    with Altman_col1:
        st.dataframe(Altman_DF_T1)
    with Altman_col2:
        st.dataframe(Altman_DF_T2)

    
elif score == 'Ohlson O-Score':
    Ohlson_DF = Stock_Year[['Ohlson_A', 'Ohlson_B', 'Ohlson_C', 'Ohlson_D',
                            'Ohlson_E', 'Ohlson_F', 'Ohlson_G', 'Ohlson_H', 
                            'Ohlson_I']]
    Ohlson_DF['O-Score'] = Ohlson_DF.apply(lambda row: OhlsonOScore(row['Ohlson_A'], row['Ohlson_B'], row['Ohlson_C'], row['Ohlson_D'], row['Ohlson_E'], 
                                                                 row['Ohlson_F'], row['Ohlson_G'], row['Ohlson_H'], row['Ohlson_I']), axis = 1)
    Ohlson_DF['Default Prob (%)'] = Ohlson_DF.apply(lambda row: Prob(row['O-Score']), axis = 1)
    Ohlson_DF.rename(columns = {'Ohlson_A':'TA/GNP', 'Ohlson_B':'TL/TA', 'Ohlson_C':'WC/TA', 
                                'Ohlson_D': 'CL/CA', 'Ohlson_E':'X', 'Ohlson_F':'NI/TA',
                                'Ohlson_G': 'FFO/TL', 'Ohlson_H':'Y', 'Ohlson_I': 'NI Ratio'}, inplace = True)
    
    Ohlson_DF = Ohlson_DF.apply(lambda s: s.apply('{0:.2f}'.format))
    Ohlson_DF["TA/GNP"] = Ohlson_DF["TA/GNP"].apply(lambda x: f"{x:.2e}")
    
    Ohlson_DF_T = Ohlson_DF.T
    Ohlson_DF_T.rename(columns= {Ohlson_DF_T.columns[0]:' '}, inplace = True)
    Ohlson_DF_T1 = Altman_DF_T.iloc[:6]
    Ohlson_DF_T2 = Altman_DF_T.iloc[6:]
    
    Ohlson_col1, Ohlson_col2 = st.columns(2)
    with Ohlson_col1:
        st.dataframe(Ohlson_DF_T1)
    with Ohlson_col2:
        st.dataframe(Ohlson_DF_T2)   
else: 
    st.print('Please Select an Credit Score Model.')
    
    
    

