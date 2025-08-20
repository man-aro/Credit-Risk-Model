score = st.selectbox("Select Iterations for KMV Merton Model: ", (10, 20 ,30, 40 ,50))

url_KMVData = "https://raw.githubusercontent.com/man-aro/Credit-Risk-Model/main/KMV_Merton/" + stock + "/KMV_Merton_Data_" + stock + "_" + str(year) + ".csv"

data = pd.read_csv(url_KMVData)

stock_summary = data['Close'].describe()

summary = stock_summary
summary = summary[['mean', 'std', '25%', '50%', '75%']].apply(lambda x: f"{x:.2f}")
summary = pd.DataFrame(summary).T
summary.rename(columns = {'mean': 'Mean', 'std':'Standard Deviation', '25%':'P25', '50%': 'P50', '75%': 'P75'}, inplace = True)
summary = summary.T
summary.rename(columns = {'Close': 'Stock Price ($)'}, inplace = True)

SD = data['Date'].tolist()[0]
ED = data['Date'].tolist()[-1]
R = (data['Rf']*100).apply(lambda x: f"{x:.2f}").tolist()[-1]
D = data['Strike_Price'].apply(lambda x: f"{x:.2e}").tolist()[-1]
MC = data['marketCapitalization'].apply(lambda x: f"{x:.2e}").tolist()[-1]

values = pd.DataFrame([SD, ED, R, D, MC]).T
values.rename(columns = {0: 'Start Date', 1:'End Date', 2: 'Rf (%)', 3:'Strike Price ($)', 4: 'Mkt Cap ($)'}, inplace = True)
values = values.T
values.rename(columns = {0: ' '}, inplace = True)

Sum_col1, Sum_col2 = st.columns(2)
with Sum_col1:
    st.dataframe(values)
with Sum_col2:
    st.dataframe(summary) 
