import streamlit as st
import yfinance as yf
from datetime import date
from prophet import Prophet
import plotly.graph_objects as go








start = "2020-01-01"
today = date.today().strftime("%Y-%m-%d")

st.title("Stock prediction app")
stocks = ["AAPL","MSFT","GOOG"]
selected_stocks = st.selectbox("Select stock for prediction",stocks)

n_years = st.slider("years of prediction:",1 , 4)
period = n_years *365 


st.cache_data
def load_data(ticker):
    data = yf.download(ticker,start,today)
    data.reset_index(inplace = True)
   
    return data


data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done")


st.subheader("Raw data")
st.write(data)

def plot():
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='Stock_Open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='Stock_Close'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot()



train = data[['Date','Close']]

train = train.rename(columns={"Date":"ds","Close":'y'})

m = Prophet()
m.fit(train)

future = m.make_future_dataframe(periods=period)

forcast = m.predict(future)


st.subheader("forcast data")

st.write(forcast)




