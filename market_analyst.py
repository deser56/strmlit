import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from talib import RSI, BBANDS, MACD
import threading
import websocket
import json

# Configure Streamlit page
st.set_page_config(page_title="Live Market Analyst", layout="wide")

# App title
st.title("📈 Real-Time Market Analysis Engine")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    time_interval = st.selectbox("Time Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])
    lookback = st.number_input("Lookback Period (Days)", 1, 365, 30)
    
    st.subheader("Technical Indicators")
    show_rsi = st.checkbox("RSI", True)
    show_macd = st.checkbox("MACD", True)
    show_bollinger = st.checkbox("Bollinger Bands", True)

# Initialize WebSocket connection
def on_message(ws, message):
    data = json.loads(message)
    # Process real-time data here
    st.session_state.latest_price = data['p']

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection established")

# Function to start WebSocket connection
def start_websocket(ticker):
    socket = f"wss://streamer.finance.yahoo.com/ws/{ticker.lower()}"
    ws = websocket.WebSocketApp(socket,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    ws.run_forever()

# Start WebSocket in background thread
if 'ws_thread' not in st.session_state:
    st.session_state.ws_thread = threading.Thread(target=start_websocket, args=(ticker,))
    st.session_state.ws_thread.daemon = True
    st.session_state.ws_thread.start()

# Fetch historical data
@st.cache_data
def get_historical_data(ticker, period, interval):
    stock = yf.Ticker(ticker)
    return stock.history(period=f"{period}d", interval=interval)

# Calculate indicators
def calculate_technical_indicators(df):
    # RSI
    df['RSI'] = RSI(df['Close'], timeperiod=14)
    
    # Bollinger Bands
    upper, middle, lower = BBANDS(df['Close'], timeperiod=20)
    df['Upper Band'] = upper
    df['Middle Band'] = middle
    df['Lower Band'] = lower
    
    # MACD
    macd, signal, hist = MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['Signal'] = signal
    df['Histogram'] = hist
    
    return df

# Generate trading signals
def generate_signals(df):
    signals = []
    
    # RSI Signals
    if df['RSI'].iloc[-1] < 30:
        signals.append(('RSI', 'BUY', df['RSI'].iloc[-1]))
    elif df['RSI'].iloc[-1] > 70:
        signals.append(('RSI', 'SELL', df['RSI'].iloc[-1]))
    
    # MACD Signals
    if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['Signal'].iloc[-2]:
        signals.append(('MACD', 'BUY', df['MACD'].iloc[-1]))
    elif df['MACD'].iloc[-1] < df['Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['Signal'].iloc[-2]:
        signals.append(('MACD', 'SELL', df['MACD'].iloc[-1]))
    
    return signals

# Create interactive chart
def create_chart(df, show_rsi, show_macd, show_bollinger):
    fig = make_subplots(rows=3 if show_rsi or show_macd else 1, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2] if show_rsi and show_macd else [0.8, 0.2])

    # Price Chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'),
                  row=1, col=1)

    # Bollinger Bands
    if show_bollinger:
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'],
                                line=dict(color='rgba(255, 0, 0, 0.5)'),
                                name='Upper Band'),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Middle Band'],
                                line=dict(color='rgba(0, 255, 0, 0.5)'),
                                name='Middle Band'),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'],
                                line=dict(color='rgba(0, 0, 255, 0.5)'),
                                name='Lower Band'),
                     row=1, col=1)

    # RSI
    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                                line=dict(color='purple'),
                                name='RSI'),
                     row=2, col=1)
        fig.add_hline(y=30, row=2, col=1, line_dash="dot", line_color="green")
        fig.add_hline(y=70, row=2, col=1, line_dash="dot", line_color="red")

    # MACD
    if show_macd:
        fig.add_trace(go.Bar(x=df.index, y=df['Histogram'],
                           name='MACD Histogram',
                           marker_color=np.where(df['Histogram'] < 0, 'red', 'green')),
                     row=3 if show_rsi else 2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],
                                line=dict(color='blue'),
                                name='MACD'),
                     row=3 if show_rsi else 2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'],
                                line=dict(color='orange'),
                                name='Signal'),
                     row=3 if show_rsi else 2, col=1)

    fig.update_layout(height=800,
                      xaxis_rangeslider_visible=False,
                      template='plotly_dark',
                      hovermode='x unified')
    
    return fig

# Main app logic
try:
    # Get data
    df = get_historical_data(ticker, lookback, time_interval)
    df = calculate_technical_indicators(df)
    
    # Generate signals
    signals = generate_signals(df)
    
    # Display latest price
    if 'latest_price' in st.session_state:
        st.markdown(f"**Latest Price:** ${st.session_state.latest_price:.2f}")
    
    # Display signals
    if signals:
        st.subheader("Trading Signals")
        cols = st.columns(len(signals))
        for col, (indicator, signal, value) in zip(cols, signals):
            color = 'green' if signal == 'BUY' else 'red'
            col.markdown(f"<h3 style='color:{color}'>{indicator} {signal}</h3>", unsafe_allow_html=True)
            col.write(f"Value: {value:.2f}")
    
    # Show chart
    st.plotly_chart(create_chart(df, show_rsi, show_macd, show_bollinger), use_container_width=True)
    
    # Data table
    with st.expander("View Raw Data"):
        st.dataframe(df.tail(20).style.background_gradient(cmap='viridis'))

except Exception as e:
    st.error(f"Error fetching data: {str(e)}")