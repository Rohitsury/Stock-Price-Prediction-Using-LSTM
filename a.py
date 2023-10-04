import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from yahoo_fin import stock_info as si
from nltk.sentiment import SentimentIntensityAnalyzer
import requests

    
st.set_page_config(layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.title('Stock Price Predictor')

user_input = st.sidebar.text_input('Enter Stock Code', 'VEDL.NS')

try:
    stock = yf.Ticker(user_input)
    stock_info = stock.info
    st.sidebar.write(f"Stock Code Entered: {user_input}")
    st.subheader(f"{stock_info['longName']}")
    stock_name = stock_info['longName']

except Exception:
    st.sidebar.error(f"Invalid stock code")
    st.stop()

yf.pdr_override()
start_date = datetime(2010, 1, 1)
start = st.sidebar.date_input('Enter Starting date', value=start_date)
import datetime
end = st.sidebar.date_input('Enter Ending date ')
end = end + datetime.timedelta(days=1)
df = pdr.get_data_yahoo(user_input, start=start, end=end)



# Function to perform sentiment analysis on text
def perform_sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']  # Return the compound sentiment score

# Fetch news articles from GNews API
def fetch_news_articles(user_input):
    url = f"https://gnews.io/api/v4/search?q={user_input}&token=YOUR_API_KEY"
    response = requests.get(url)
    data = response.json()
    
    if 'articles' in data:
        articles = data['articles']
    elif 'error' in data:
        print(f"API Error: {data['error']}")
        articles = []
    else:
        print("Unknown response format")
        articles = []
    
    return articles


st.write(len(df))
if len(df) <= 100:
    st.sidebar.error("Enter Minimum 101 working Days")
    st.stop()

if st.sidebar.button("Get Predicted Value"):
    st.subheader(f'Data From {start} - {end}')
    first, last = st.columns(2)

    first.write("Initial Rows")
    last.write("Ending Rows")
    first.write(df.head())
    last.write(df.tail())

    first.subheader("Opening price")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.index, df.Open)
    plt.xlabel("Date")
    plt.ylabel("Price")
    first.pyplot(fig)

    last.subheader("closing price")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.index, df.Close)
    plt.xlabel("Date")
    plt.ylabel("Price")
    last.pyplot(fig)

    df = pd.DataFrame(df['Close'])
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    x_train = []
    y_train = []

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    model = load_model('LSTM_model.h5')

    x_test = data_testing
    y_test = data_testing

    x_test_scaled = scaler.transform(x_test)

    x_test = []
    y_test = []

    for i in range(100, x_test_scaled.shape[0]):
        x_test.append(x_test_scaled[i-100:i])
        y_test.append(x_test_scaled[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    y_predicted = scaler.inverse_transform(y_predicted)
    y_test = scaler.inverse_transform([y_test])

    st.subheader("Predicted vs Original Values")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test[0], 'b', label="Original Values")
    plt.plot(y_predicted[:, 0], 'r', label="Predicted Values")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    articles = fetch_news_articles(user_input)
    sentiment_scores = [perform_sentiment_analysis(article['title']) for article in articles]

    st.subheader("News Sentiment Analysis")
    for i, article in enumerate(articles):
        st.write(f"Title: {article['title']}")
        st.write(f"Published At: {article['publishedAt']}")
        st.write(f"Sentiment Score: {sentiment_scores[i]}")
        st.write("-----")

    lstm_pred = "{:.2f}".format(y_predicted[-1][0])

    c1, c2 = st.columns(2)
    c1.info("LSTM")
    c1.info(f"Closing Price: {lstm_pred}")

    from sklearn.metrics import mean_absolute_percentage_error

    mape = mean_absolute_percentage_error(y_test[0], y_predicted[:, 0])
    mape = 100 - (mape * 100)

    c1.info(f"**Accuracy: {mape}**")
