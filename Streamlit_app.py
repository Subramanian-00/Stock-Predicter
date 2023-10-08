import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2023-7-30'

st.title('Stock Future Predicter')

use_input = st.text_input('Enter stock Ticker', 'AAPL')  ##############

if st.button('Predict'):
    df = yf.download(use_input, start, end)

    # describing data
    st.subheader('Data From 2010-2023')
    st.write(df.describe())

    # maps

    st.subheader('closing Price VS Time Chart ')
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df.Close, color='yellow')
    plt.legend()
    st.pyplot(fig)

    st.subheader('closing Price VS Time Chart with 100 moving Average  ')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(ma100, color='red')
    plt.plot(df.Close, color='yellow')
    plt.legend()
    st.pyplot(fig)

    st.subheader('closing Price VS Time Chart with 100 & 200 moving Average  ')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(ma100, color='red')
    plt.plot(ma200, color='green')
    plt.plot(df.Close, color='yellow')
    plt.legend()
    st.pyplot(fig)

    # spltting data into train test
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    print(' taining ', data_training.shape)
    print(' testing ', data_testing.shape)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))

    data_training_array = scaler.fit_transform(data_training)
