
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="BTC/USD Dashboard",
    page_icon=":money_with_wings:",
    layout="wide"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "image.jpg").replace("\\", "/")

if os.path.exists(image_path):
    page_bg_img = f"""
    <style>
    body {{
        background-image: url("{image_path}");
        background-size: cover;
        background-attachment: fixed;
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.6);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("📈 Bitcoin (BTC) USD Dashboard")
st.markdown("Interactive cryptocurrency analysis with Plotly and ARIMA forecasting")


@st.cache_data(ttl=3600)  
def load_live_data(ticker="BTC-USD", period="2y"):
    df = yf.download(ticker, period=period)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = load_live_data()
st.sidebar.header("Options")

chart_type = st.sidebar.selectbox(
    "Choose chart type:",
    ["Line Chart", "Candlestick Chart", "ARIMA Forecast"]
)

start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())

if start_date > end_date:
    st.sidebar.error("Start Date must be before End Date!")


mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
data_filtered = df.loc[mask].reset_index(drop=True)

if st.sidebar.button("Refresh Data"):
    df = load_live_data()
    data_filtered = df.loc[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))].reset_index(drop=True)
    st.success("Data updated!")
if chart_type == "Line Chart":
    st.subheader("BTC/USD Closing Price Over Time")
    fig = px.line(
        data_filtered,
        x='Date',
        y='Close',
        title='BTC/USD Closing Price',
        labels={'Close':'Price (USD)'},
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
elif chart_type == "Candlestick Chart":
    st.subheader("Candlestick Chart")
    
    fig = go.Figure(
        data=[go.Candlestick(
            x=data_filtered['Date'],
            open=data_filtered['Open'],
            high=data_filtered['High'],
            low=data_filtered['Low'],
            close=data_filtered['Close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )]
    )
    fig.add_trace(go.Bar(
        x=data_filtered['Date'],
        y=data_filtered['Volume'],
        name='Volume',
        marker_color='blue',
        yaxis='y2',
        opacity=0.3
    ))

    fig.update_layout(
        title='BTC/USD Candlestick with Volume',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template="plotly_dark",
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False,
            title='Volume'
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "ARIMA Forecast":
    st.subheader("ARIMA Forecast")
    st.info("Forecasting next 30 days of BTC/USD closing price")

    close_prices = data_filtered['Close'].reset_index(drop=True)

    if len(close_prices) < 30:
        st.warning("Not enough data for ARIMA forecast. Select a larger date range.")
    else:
        
        model = ARIMA(close_prices, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)
        forecast_dates = pd.date_range(data_filtered['Date'].max() + pd.Timedelta(days=1), periods=30)

        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_filtered['Date'], y=data_filtered['Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast',
                                 line=dict(dash='dash', color='orange')))

        fig.update_layout(
            title='BTC/USD ARIMA Forecast',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(data_filtered)
