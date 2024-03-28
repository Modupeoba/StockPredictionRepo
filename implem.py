
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')
import joblib 

import warnings 
warnings.filterwarnings('ignore')


st.title(" :bar_chart: StockSense: Enhancing Investment Decisions With Predictive Stock Price Modeling")
st.markdown("<br>", unsafe_allow_html= True)

st.image('cardano-blockchain-platform-collage.png')

st.sidebar.image('3112677-removebg-preview.png', width = 300, caption = 'Welcome User')
st.sidebar.divider()
st.sidebar.markdown("<br>", unsafe_allow_html= True)


ticks = ['AAPL (Apple Inc.), MSFT (Microsoft Corporation),AMZN (Amazon.com, Inc.),TSLA (Tesla, Inc.),XOM (ExxonMobil Corporation),JPM (JPMorgan Chase & Co.),V (Visa Inc.),GE (General Electric Company),MON (Monsanto Company),INTC (Intel Corporation),ACB (Canadian Marijuana Companies),AMC (AMC Entertainment Holdings, Inc.),Zoom (Zoom Video Communications, Inc.),TRIP (TripAdvisor Worldwide, Inc.),UZNH (United States Natural Gas Holdings, Inc.),SPE (Royal Dutch Shell A Ltd),Fs(First State Investments),XNJ (ExxonMobil Chemicals Company),BRK-A (Berkshire Hathaway Inc.),LQD (First State Telstra Limited),CSCO (Cisco Systems, Inc.),EXC (ExxonMobil Corporation),WMT (Walmart Inc.),JNJ (Johnson & Johnson),IBM (International Business Machines Corporation),CVS (CVS Health Corporation),DVV (Daimler AG),MNDP (Medtronic, Inc.)']
ticks2 = []
for i in ticks:
    ticks2.append(i.split(','))
ticks2 = ticks2[0]


st.divider()
ticker = st.selectbox('Tickers List', ticks2)
ticker = ticker.split('(')[0]

start = st.date_input('Select The Start State')
end = st.date_input('Select The Stop Date')

def data_collector(ticker_symbol):
    stock_data = yf.download(ticker_symbol, start = start, end = end)
    return stock_data

data = data_collector(ticker)

st.divider()
st.subheader(f'{ticker} Data From {start} To {end}')

st.dataframe(data)

ds = data.copy()
ds.reset_index(inplace = True)
ds.index = pd.to_datetime(ds.Date)
ds = ds[['Close']]


# Convert the dataframe to a numpy array
dx = ds.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dx) * .70 ))


# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dx)



# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape


from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=5)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dx[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))



# Plot the data
sns.set(style = 'darkgrid')

train = ds[:training_data_len]
valid = ds[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
# plt.figure(figsize=(16,6))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

import plotly.express as px
import plotly.graph_objs as go

fig = go.Figure()

# Adding train data
fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))

# Adding validation data and predictions
fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Validation'))
fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))

# Customizing layout
fig.update_layout(
    title='Model',
    xaxis=dict(title='Date', tickformat='%Y-%m-%d'),
    yaxis=dict(title='Close Price USD ($)'),
    legend=dict(x=0.05, y=0.95),
    autosize=False,
    width=800,
    height=500,
)

# Displaying the plot
st.plotly_chart(fig)

# from keras.models import load_model
# loaded_model = load_model("lstm_timeSeries.keras")

