# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 21:24:01 2025

@author: ManishArora
"""

import streamlit as st
import pandas as pd

st.title('Credit Risk Models')

url_CreditModel = "https://github.com/man-aro/Credit-Risk-Model/blob/main/Dataset_Relevant_CreditModel.csv"
url_KMVData = "https://github.com/man-aro/Credit-Risk-Model/tree/main/KMV_Merton."

Credit_Model = pd.read_excel(url_CreditModel)


st.write("### Preview of Data")