import streamlit as st
import pickle
import pandas as pd

st.title("Parkinson's disease prediction from speech features")
st.info("Application for Parkinson's disease prediction from speech features")
st.sidebar.header('Speech feature selection:')

gender=st.text_input('gender')
tqwt_TKEO_mean_dec_11=st.text_input('tqwt_TKEO_mean_dec_11')
tqwt_TKEO_mean_dec_16=st.text_input('tqwt_TKEO_mean_dec_16')
tqwt_TKEO_std_dec_11=st.text_input('tqwt_TKEO_std_dec_11')
tqwt_TKEO_std_dec_12=st.text_input('tqwt_TKEO_std_dec_12')
tqwt_TKEO_std_dec_15=st.text_input('tqwt_TKEO_std_dec_15')
tqwt_energy_dec_12=st.text_input('tqwt_energy_dec_12')
tqwt_energy_dec_13=st.text_input('tqwt_energy_dec_13')
tqwt_energy_dec_15=st.text_input('tqwt_energy_dec_15')
tqwt_energy_dec_27=st.text_input('tqwt_energy_dec_27')
tqwt_entropy_shannon_dec_10=st.text_input('tqwt_entropy_shannon_dec_10')
tqwt_entropy_shannon_dec_11=st.text_input('tqwt_entropy_shannon_dec_11')
tqwt_entropy_shannon_dec_12=st.text_input('tqwt_entropy_shannon_dec_12')
tqwt_entropy_shannon_dec_13=st.text_input('tqwt_entropy_shannon_dec_13')
tqwt_entropy_shannon_dec_14=st.text_input('tqwt_entropy_shannon_dec_14')
tqwt_entropy_shannon_dec_15=st.text_input('tqwt_entropy_shannon_dec_15')
tqwt_entropy_shannon_dec_16=st.text_input('tqwt_entropy_shannon_dec_16')
tqwt_entropy_shannon_dec_34=st.text_input('tqwt_entropy_shannon_dec_34')
tqwt_kurtosisValue_dec_25=st.text_input('tqwt_kurtosisValue_dec_25')
tqwt_kurtosisValue_dec_26=st.text_input('tqwt_kurtosisValue_dec_26')
tqwt_kurtosisValue_dec_27=st.text_input('tqwt_kurtosisValue_dec_27')
tqwt_kurtosisValue_dec_28=st.text_input('tqwt_kurtosisValue_dec_28')
tqwt_kurtosisValue_dec_36=st.text_input('tqwt_kurtosisValue_dec_36')
tqwt_maxValue_dec_11=st.text_input('tqwt_maxValue_dec_11')
tqwt_maxValue_dec_12=st.text_input('tqwt_maxValue_dec_12')
tqwt_maxValue_dec_13=st.text_input('tqwt_maxValue_dec_13')
tqwt_stdValue_dec_11=st.text_input('tqwt_stdValue_dec_11')
tqwt_stdValue_dec_12=st.text_input('tqwt_stdValue_dec_12')
tqwt_stdValue_dec_15=st.text_input('tqwt_stdValue_dec_15')
tqwt_stdValue_dec_16=st.text_input('tqwt_stdValue_dec_16')

df=pd.DataFrame({'gender':[gender],'tqwt_TKEO_mean_dec_11':[tqwt_TKEO_mean_dec_11],
                 'tqwt_TKEO_mean_dec_16':[tqwt_TKEO_mean_dec_16],
                 'tqwt_TKEO_std_dec_11':[tqwt_TKEO_std_dec_11],
                 'tqwt_TKEO_std_dec_12':[tqwt_TKEO_std_dec_12],
                 'tqwt_TKEO_std_dec_15':[tqwt_TKEO_std_dec_15],
                 'tqwt_energy_dec_27':[tqwt_energy_dec_27],
                 'tqwt_entropy_shannon_dec_10':[tqwt_entropy_shannon_dec_10],
                 'tqwt_entropy_shannon_dec_11':[tqwt_entropy_shannon_dec_11],
                 'tqwt_entropy_shannon_dec_12':[tqwt_entropy_shannon_dec_12],
                 'tqwt_entropy_shannon_dec_13':[tqwt_entropy_shannon_dec_13],
                 'tqwt_entropy_shannon_dec_14':[tqwt_entropy_shannon_dec_14],
                 'tqwt_entropy_shannon_dec_15':[tqwt_entropy_shannon_dec_15],
                 'tqwt_entropy_shannon_dec_16':[tqwt_entropy_shannon_dec_16],
                 'tqwt_entropy_shannon_dec_34':[tqwt_entropy_shannon_dec_34],
                 'tqwt_kurtosisValue_dec_25':[tqwt_kurtosisValue_dec_25],
                 'tqwt_kurtosisValue_dec_26':[tqwt_kurtosisValue_dec_26],
                 'tqwt_kurtosisValue_dec_27':[tqwt_kurtosisValue_dec_27],
                 'tqwt_kurtosisValue_dec_28':[tqwt_kurtosisValue_dec_28],
                 'tqwt_kurtosisValue_dec_36':[tqwt_kurtosisValue_dec_36],
                 'tqwt_maxValue_dec_11':[tqwt_maxValue_dec_11],
                 'tqwt_maxValue_dec_12':[tqwt_maxValue_dec_12],
                 'tqwt_maxValue_dec_13':[tqwt_maxValue_dec_13],
                 'tqwt_stdValue_dec_11':[tqwt_stdValue_dec_11],
                 'tqwt_stdValue_dec_12':[tqwt_stdValue_dec_12],
                 'tqwt_stdValue_dec_15':[tqwt_stdValue_dec_15],
                 'tqwt_stdValue_dec_16':[tqwt_stdValue_dec_16]}, index=[0])

model=pickle.load(open(r'C:\Users\Nourine\Desktop\Parkinsons prediction\pd_speech_features.sav','rb'))

Con=st.sidebar.button('confirm')
if Con:
    result=model.predict(df)
    if result == 0:
        st.sidebar.write("Patient might not have Parkinson's disease")
    else:
        st.sidebar.write("Patient likely to have Parkinson's disease")

