import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
# from datetime import datetime
from datetime import datetime
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from yahoo_fin import stock_info as si


# from sklearn.metrics import r2_score
st.set_page_config(layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Title and Stock ticker
st.sidebar.title('Stock Price Predictor')

user_input = st.sidebar.text_input('Enter Stock Code','VEDL.NS')

 
st.sidebar.write(f"Stock Code Entered : {user_input}")
 
 
        


# Fetching Data From Yahoo Finanace 
yf.pdr_override()
start_date = datetime(2010, 1, 1)
start = st.sidebar.date_input('Enter Starting date',  value=start_date)
import datetime
end = st.sidebar.date_input('Enter Ending date ')
end = end + datetime.timedelta(days=1)
df = pdr.get_data_yahoo(user_input, start=start,end=end)

st.write(len(df))
if(len(df) <= 100):
    st.sidebar.error("Enter Minimum 101 working Days")
    st.stop()

if st.sidebar.button("Get Predicted Value"):
    st.subheader(f'Data From {start} - {end}')
    first, last = st.columns(2)
    # Describe data

    first.write("Initial Rows")
    last.write("Ending Rows")
    first.write(df.head())
    last.write(df.tail())

    first.subheader("Opening price")
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.index,df.Open)
    plt.xlabel("Date")
    plt.ylabel("Price")
    first.pyplot(fig)

    last.subheader("closing price")
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.index,df.Close)
    plt.xlabel("Date")
    plt.ylabel("Price")
    last.pyplot(fig)

    # last.subheader("closing price vs Time Chart with 100MA")
    # ma100 = df.Close.rolling(100).mean()
    # fig = plt.figure(figsize=(12,6))
    # plt.plot(ma100)
    # plt.plot(df.Close)
    # last.pyplot(fig)

    # st.subheader("closing price vs Time Chart with 100MA & 200MA")
    # ma100 = df.Close.rolling(100).mean()
    # ma200 = df.Close.rolling(200).mean()
    # fig = plt.figure(figsize=(12,6))
    # plt.plot(ma100)
    # plt.plot(ma200)
    # plt.plot(df.Close)
    # st.pyplot(fig)

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])


    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)

    # Load Model

    model = load_model('LSTM_model.h5')

    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index = True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range (100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test,y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    scaler = scaler.scale_
    scaler_factor = 1/scaler[0]
    y_predicted = y_predicted * scaler_factor
    y_test = y_test * scaler_factor

    st.subheader("Predicted vs Original Values")
    fig2= plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = "Original Values")
    plt.plot(y_predicted , 'r' ,label = "PredictedValues")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    # import datetime
    # next day
    yf.pdr_override()
    # start = datetime.datetime.now() - datetime.timedelta(days=50) ,datetime.datetime.now()
    stock = pdr.get_data_yahoo(user_input, start=start,end=end)

    new_df=stock.filter(['Close'])

    scaler = MinMaxScaler(feature_range=(0,1))
    last_100_days=new_df[-int(len(new_df)):].values
    last_100_days_scaled=scaler.fit_transform(last_100_days)

    X_test=[]
    X_test.append(last_100_days_scaled)

    X_test=np.array(X_test)

    #Get the predicted scaled price
    pred_price=model.predict(X_test)
    #undo the scaling
    pred_price=scaler.inverse_transform(pred_price)
    pred_price = float(pred_price)

    # pred_day=datetime.datetime.now() + datetime.timedelta(days=1)
    lstm_pred = "{:.2f}".format(pred_price)
    # st.subheader(f'The Next Predicted Closing Price for {stock_name}({user_input}) is')
    
    c1,c2 = st.columns(2)
    c1.info("LSTM")
    c1.info(f"Closing Price : {lstm_pred}")
        
    # assuming y_test and y_predicted are the predicted and actual values
    # r2_score_lstm = r2_score(y_test, y_predicted)
    # r2_score_lstm = r2_score_lstm * 100
    # c1.write(f"Accuracy : {r2_score_lstm}")
    
    from sklearn.metrics import mean_absolute_percentage_error

    # mse = mean_squared_error(y_test, y_predicted)
    # rmse = np.sqrt(mse)
    # mae = mean_absolute_error(y_test, y_predicted)

    mape = mean_absolute_percentage_error(y_test, y_predicted)
    mape = 100 - (mape * 100) 

    # c1.warning(f"MSE: {mse}")
    # c1.warning(f"RMSE: {rmse}")
    # c1.warning(f"MAE: {mae}")
    c1.info(f"**Accuracy: {mape}**")


    # Linear   Regression And Decision Tree Model

    # from sklearn.tree import DecisionTreeRegressor
    # from sklearn.linear_model import LinearRegression
    # from sklearn.model_selection import train_test_split

    # df2 = df['Close']
    # df2 = pd.DataFrame(df2)   

    # Prediction 100 days into the future.
    # future_days = 1
    # df2['Prediction_Close'] = df2['Close'].shift(-future_days)
    # x_close = np.array(df2.drop(['Prediction_Close'], 1))[:-future_days]
    # y_close = np.array(df2['Prediction_Close'])[:-future_days]
    # x_close_train, x_close_test, y_close_train, y_close_test = train_test_split(x_close, y_close, test_size = 0.2)

    # Implementing Linear and Decision Tree Regression Algorithms.

    # tree_close = DecisionTreeRegressor().fit(x_close_train, y_close_train)
    # lr_close = LinearRegression().fit(x_close_train, y_close_train)

    # x_future_close = df2.drop(['Prediction_Close'], 1)[:-future_days]
    # x_future_close = x_future_close.tail(future_days)
    # x_future_close = np.array(x_future_close)

    # tree_prediction_close = tree_close.predict(x_future_close)
    # tree_prediction_close = float(tree_prediction_close)
    # tree_prediction_close = "{:.2f}".format(tree_prediction_close)

    # c2.write(f"Decision Tree")
    # c2.write(f"Closing Price  {tree_prediction_close}")

    # lr_prediction_close = lr_close.predict(x_future_close)
    # lr_prediction_close = float(lr_prediction_close)
    # lr_prediction_close = "{:.2f}".format(lr_prediction_close)
    # c2.write(f"Linear Regression")
    # c2.write(f"Closing Price  {lr_prediction_close}")

    # Compute R^2 score for linear regression
    # lr_score = lr_close.score(x_close_test, y_close_test)
    # lr_score =lr_score * 100
    # c3.write("Linear regression R^2 score: {:.2f}".format(lr_score))

    # Compute R^2 score for decision tree
    # tree_score = tree_close.score(x_close_test, y_close_test)
    # tree_score = tree_score * 100
    # c2.write("Decision tree R^2 score: {:.2f}".format(tree_score))

    # from sklearn.model_selection import cross_val_score
    # s = cross_val_score(lr_close, x_close_test, y_close_test, cv=10)


    # st.write("Cross-validation scores: {}".format(s))
    # st.write("Average cross-validation score: {:.2f}".format(s.mean()))
