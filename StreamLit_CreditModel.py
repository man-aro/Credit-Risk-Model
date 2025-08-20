# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 21:24:01 2025

@author: ManishArora
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

st.title('Credit Risk Models')
st.write('Author: Manish Rajkumar Arora')
st.write("DataSources: FRED, YahooFinance, FMP.")

url_CreditModel = "https://raw.githubusercontent.com/man-aro/Credit-Risk-Model/main/Dataset_Relevant_CreditModel.csv"


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

score = st.selectbox("Credit Model: ", ('Altman Z-Score', 'Ohlson O-Score', 'KMV-Merton'))

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
    
    Ohlson_DF_Format = Ohlson_DF.apply(lambda s: s.apply('{0:.2f}'.format))
    Ohlson_DF_Format["TA/GNP"] = Ohlson_DF["TA/GNP"].apply(lambda x: f"{x:.2e}")
    
    Ohlson_DF_T = Ohlson_DF_Format.T
    Ohlson_DF_T.rename(columns= {Ohlson_DF_T.columns[0]:' '}, inplace = True)
    Ohlson_DF_T1 = Ohlson_DF_T.iloc[:6]
    Ohlson_DF_T2 = Ohlson_DF_T.iloc[6:]
    
    Ohlson_col1, Ohlson_col2 = st.columns(2)
    with Ohlson_col1:
        st.dataframe(Ohlson_DF_T1)
    with Ohlson_col2:
        st.dataframe(Ohlson_DF_T2)   
elif score == 'KMV-Merton': 
    
    url_KMVData = "https://raw.githubusercontent.com/man-aro/Credit-Risk-Model/main/KMV_Merton/" + stock + "/KMV_Merton_Data_" + stock + "_" + str(year) + ".csv"
    
    KMVData = pd.read_csv(url_KMVData)
    
    stock_summary = KMVData['Close'].describe()
    
    summary = stock_summary
    summary = summary[['mean', 'std', '25%', '50%', '75%']].apply(lambda x: f"{x:.2f}")
    summary = pd.DataFrame(summary).T
    summary.rename(columns = {'mean': 'Mean', 'std':'SD', '25%':'P25', '50%': 'P50', '75%': 'P75'}, inplace = True)
    summary = summary.T
    summary.rename(columns = {'Close': 'Stock Price ($)'}, inplace = True)
    
    SD = KMVData['Date'].tolist()[0]
    ED = KMVData['Date'].tolist()[-1]
    R = (KMVData['Rf']*100).apply(lambda x: f"{x:.2f}").tolist()[-1]
    D = KMVData['Strike_Price'].apply(lambda x: f"{x:.2e}").tolist()[-1]
    MC = KMVData['marketCapitalization'].apply(lambda x: f"{x:.2e}").tolist()[-1]
    
    values = pd.DataFrame([SD, ED, R, D, MC]).T
    values.rename(columns = {0: 'Start Date', 1:'End Date', 2: 'Rf (%)', 3:'Strike Price ($)', 4: 'Mkt Cap ($)'}, inplace = True)
    values = values.T
    values.rename(columns = {0: ' '}, inplace = True)
    
    Sum_col1, Sum_col2 = st.columns(2)
    with Sum_col1:
        st.dataframe(values)
    with Sum_col2:
        st.dataframe(summary) 
    
    N = len(KMVData) 
    MktCap = KMVData['marketCapitalization'].unique()[0] #Current Market Cap
    NumShares = KMVData['numberOfShares'].unique()[0] 
    Equity = (KMVData['Close']*NumShares).values #Market Cap timeseries (historical)
    
    #Strike Price
    Strike_Price = KMVData['Strike_Price'].unique()[0]
    
    #Volatility
    Annual_Stock_Volatility = np.std(KMVData['Returns'])*np.sqrt(N+1)
    Initial_Volatility = (Annual_Stock_Volatility*MktCap)/(MktCap + Strike_Price) #Vol in terms of Market Capital
    
    #Interest Rate: Assumed constant (current rate)
    Rate = KMVData[KMVData['Date'] == KMVData['Date'].max()]['Rf'].values
    
    #Maturity
    T = 1 #Maturity (approximately 1)
    
    M = 50
    Sigma_Tt = [Initial_Volatility]
    Annual_MU = []
    for m in range(1, M):
        AI = (Rate + 0.5*Sigma_Tt[m-1]**2)*T
        BI = (Rate - 0.5*Sigma_Tt[m-1]**2)*T
        CI = Sigma_Tt[m-1]*np.sqrt(T)
        DI = Strike_Price*np.exp(-Rate*T)
        
        def function(x, Equity):
            d1 = (np.log(x/Strike_Price) + AI) / CI
            d2 = (np.log(x/Strike_Price) + BI) / CI
            return (x*norm.cdf(d1) - DI*norm.cdf(d2)) - Equity
    
        X = np.zeros(N)
        for s in range(N):
            sol = fsolve(lambda Asset: function(Asset, Equity[s]), x0 = MktCap)
            X[s] = sol[0]
        
        daily_mu = (1/(N-1)) * np.log(X[-1]/X[0])
        annual_mu = daily_mu/(1/N)
        Annual_MU.append(annual_mu)
        
        Vol = np.zeros(N-1)
        for v in range(len(Vol)):
            Vol[v] = (np.log(X[v+1]/X[v]) - daily_mu)**2
        sigma_tt_new = np.sqrt(np.sum(Vol)/(N-1)) * np.sqrt(N)
        Sigma_Tt.append(sigma_tt_new)
        
        if abs(Sigma_Tt[-2] - Sigma_Tt[-1]) <= 1e-5:
            break
        
    Sigma = Sigma_Tt[-1]
    Annual_Sigma = f"{Sigma:.2f}"
    A_MU = Annual_MU[-1]
    Annual_Drift = f"{A_MU:.2f}"
    Initial_Asset_Value = X[-1]
    IVA = f"{Initial_Asset_Value:.2e}"
    
    PD_A = np.log(Initial_Asset_Value/Strike_Price)
    PD_B = (A_MU - 0.5*Sigma**2)*T #d2
    PD_C = Sigma*np.sqrt(T)
    
    PD_Merton = norm.cdf(-(PD_A + PD_B)/PD_C) * 100
    ProbDef_Merton = f"{PD_Merton:.2f}"
    
    st.write("Optimisation Iterations = 50, Monte Carlo Simulations = 5000, Time Steps = 252.")
    
    def MonteCarlo(N, M, V0, mu, sigma, T, Strike_Price):
        dt = T/N
        dw = np.random.normal(0, 1, size = [N, M])
        V = np.ones([N, M])
        V[0] = V0
        time = np.zeros([N,M])
        time[0] = 0
        for i in range(1,N):
            V[i] = V[i-1,:]*np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*dw[i-1,:])
            time[i] = i*dt
        
        Prob_Default = ((V[-1] < Strike_Price).sum()/M) * 100
        return Prob_Default, V, time

    N = 253
    M_MC = 5000

    KMV_Prob_Default = MonteCarlo(N, M_MC, Initial_Asset_Value, A_MU, Sigma, T, Strike_Price)[0]
    KMV_Prob = f"{KMV_Prob_Default:.2f}"

    V = MonteCarlo(N, M_MC, Initial_Asset_Value, A_MU, Sigma, T, Strike_Price)[1]
    time = MonteCarlo(N, M_MC, Initial_Asset_Value, A_MU, Sigma, T, Strike_Price)[2]
  
    default_mask = V[-1] < Strike_Price
    survive_mask = V[-1] >= Strike_Price

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, V[:, survive_mask], color="orange", alpha=0.05)
    ax.plot(time, V[:, default_mask], color="black", alpha=0.1)
    ax.axhline(Strike_Price, color="red", linestyle="--", label="Default Barrier")
    ax.set_xlabel("Time", fontsize=15)
    ax.set_title(f"Asset Value Distribution: {stock} ({year})", fontsize=15)

    st.pyplot(fig)


    KMV_Results = pd.DataFrame([Annual_Drift, Annual_Sigma, IVA, ProbDef_Merton, KMV_Prob]).T
    KMV_Results.rename(columns = {0:'Annual Drift', 1:'Annual Sigma', 2:'Asset Value', 3:'Default Prob (%)', 4: 'MC Default Prob (%)'}, inplace = True)
    KMV_Results = KMV_Results.T
    KMV_Results.rename(columns = {0: ' '}, inplace = True)
    
    KMV_1 = KMV_Results.iloc[:3]
    KMV_2 = KMV_Results.iloc[3:]
    
    KMV_col1, KMV_col2 = st.columns(2)
    with KMV_col1:
        st.dataframe(KMV_1)
    with KMV_col2:
        st.dataframe(KMV_2) 
   
else:
    st.print('Please Select an Credit Score Model.')
    


