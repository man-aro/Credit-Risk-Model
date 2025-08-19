
if score == 'Altman Z-Score':
    Altman_DF = Stock_Year[['Altman_A', 'Altman_B', 'Altman_C', 'Altman_D',
                            'Altman_E']]
    Altman_DF['Z-Score'] = Altman_DF.apply(lambda row: AltmanZScore(row['Altman_A'], row['Altman_B'], 
                                                                 row['Altman_C'], row['Altman_D'], 
                                                                 row['Altman_E']), axis = 1)
    Altman_DF.rename(columns = {'Altman_A': 'WC/TA', 'Altman_B':'RE/TA', 'Altman_C':'EBIT/TA', 
                                'Altman_D': 'MktCap/TL', 'Altman_E':'Sales/TA'}, inplace = True)
    
    Altman_DF_Display = Altman_DF.apply(lambda s: s.apply('{0:.2e}'.format))
    st.dataframe(Altman_DF_Display.style.hide(axis="index"), use_container_width = True)
    
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
    Ohlson_DF_Display = Ohlson_DF.apply(lambda s: s.apply('{0:.2e}'.format))
    
    st.dataframe(Ohlson_DF_Display.style.hide(axis="index"), use_container_width = True)    
        
else: 
    st.print('Please Select an Credit Score Model.')
