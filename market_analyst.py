import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import websocket
import json
import threading

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

# WebSocket implementation
def on_message(ws, message):
    try:
        data = json.loads(message)
        if 'p' in data:
            st.session_state.latest_price = data['p']
    except Exception as e:
        st.error(f"WebSocket error: {str(e)}")

def start_websocket(ticker):
    ws = websocket.WebSocketApp(
        f"wss://streamer.finance.yahoo.com/ws/{ticker.lower()}",
        on_message=on_message,
        on_error=lambda ws, err: st.error(f"WebSocket error: {str(err)}"),
        on_close=lambda ws: st.warning("WebSocket connection closed")
    )
    ws.run_forever()

# Initialize WebSocket in background
if 'ws_thread' not in st.session_state:
    st.session_state.ws_thread = threading.Thread(target=start_websocket, args=(ticker,), daemon=True)
    st.session_state.ws_thread.start()

# Enhanced data fetching with error handling
@st.cache_data
def get_historical_data(ticker, period, interval):
    try:
        return yf.Ticker(ticker).history(period=f"{period}d", interval=interval)
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return pd.DataFrame()

# Technical indicator calculations using pandas
def calculate_technical_indicators(df):
    if df.empty:
        return df
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['Middle Band'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['Upper Band'] = df['Middle Band'] + (std_dev * 2)
    df['Lower Band'] = df['Middle Band'] - (std_dev * 2)
    
    # MACD Calculation
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    
    return df

# Signal generation logic
def generate_signals(df):
    signals = []
    
    # RSI Signals
    if not df.empty and 'RSI' in df.columns:
        last_rsi = df['RSI'].iloc[-1]
        if last_rsi < 30:
            signals.append(('RSI', 'BUY', last_rsi))
        elif last_rsi > 70:
            signals.append(('RSI', 'SELL', last_rsi))
    
    # MACD Signals
    if not df.empty and 'MACD' in df.columns and 'Signal' in df.columns:
        if len(df) >= 2:
            last_macd = df['MACD'].iloc[-1]
            last_signal = df['Signal'].iloc[-1]
            prev_macd = df['MACD'].iloc[-2]
            prev_signal = df['Signal'].iloc[-2]
            
            if last_macd > last_signal and prev_macd <= prev_signal:
                signals.append(('MACD', 'BUY', last_macd))
            elif last_macd < last_signal and prev_macd >= prev_signal:
                signals.append(('MACD', 'SELL', last_macd))
    
    return signals

# Enhanced chart visualization
def create_chart(df, show_rsi, show_macd, show_bollinger):
    fig = make_subplots(rows=3 if show_rsi and show_macd else 2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2] if show_rsi and show_macd else [0.8, 0.2])
    
    # Price Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#2ECC71',
        decreasing_line_color='#E74C3C'
    ), row=1, col=1)
    
    # Bollinger Bands
    if show_bollinger:
        for band, color in zip(['Upper Band', 'Middle Band', 'Lower Band'], 
                             ['#3498DB', '#F1C40F', '#E74C3C']):
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[band],
                line=dict(color=color, width=1.5),
                name=band
            ), row=1, col=1)
    
    # RSI
    if show_rsi:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            line=dict(color='#9B59B6', width=2),
            name='RSI'
        ), row=2, col=1)
        fig.add_hline(y=30, row=2, col=1, line_dash="dot", line_color="#2ECC71")
        fig.add_hline(y=70, row=2, col=1, line_dash="dot", line_color="#E74C3C")
    
    # MACD
    if show_macd:
        row_position = 3 if show_rsi else 2
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Histogram'],
            name='Histogram',
            marker_color=np.where(df['Histogram'] < 0, '#E74C3C', '#2ECC71')
        ), row=row_position, col=1)
        
        for line, color in zip(['MACD', 'Signal'], ['#3498DB', '#F1C40F']):
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[line],
                line=dict(color=color, width=1.5),
                name=line
            ), row=row_position, col=1)
    
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        hovermode='x unified',
        margin=dict(t=40, b=40)
    )
    
    return fig

# Main application logic
try:
    df = get_historical_data(ticker, lookback, time_interval)
    if not df.empty:
        df = calculate_technical_indicators(df)
        signals = generate_signals(df)
        
        # Display real-time information
        col1, col2 = st.columns(2)
        with col1:
            if 'latest_price' in st.session_state:
                st.metric(f"Current {ticker} Price", 
                          f"${st.session_state.latest_price:.2f}",
                          delta=f"{df['Close'].iloc[-1] - df['Close'].iloc[-2]:.2f}")
        
        with col2:
            if signals:
                st.success("Active Trading Signals Detected")
            else:
                st.info("No Strong Signals Currently")
        
        # Display signals
        if signals:
            with st.expander("Detailed Trading Signals", expanded=True):
                for indicator, signal, value in signals:
                    st.write(f"**{indicator}** ({value:.2f}): {signal} signal")
        
        # Display chart
        st.plotly_chart(create_chart(df, show_rsi, show_macd, show_bollinger), 
                       use_container_width=True)
        
        # Data summary
        with st.expander("Technical Summary"):
            cols = st.columns(4)
            metrics = {
                'RSI': df['RSI'].iloc[-1],
                'MACD': df['MACD'].iloc[-1],
                'Upper Band': df['Upper Band'].iloc[-1],
                'Lower Band': df['Lower Band'].iloc[-1]
            }
            for col, (metric, value) in zip(cols, metrics.items()):
                col.metric(metric, f"{value:.2f}")
                
        # Raw data
        with st.expander("Historical Data Preview"):
            st.dataframe(df.tail(10).style.format({
                'Open': '{:.2f}',
                'High': '{:.2f}',
                'Low': '{:.2f}',
                'Close': '{:.2f}',
                'RSI': '{:.2f}',
                'MACD': '{:.2f}',
                'Signal': '{:.2f}'
            }).background_gradient(cmap='viridis'))

except Exception as e:
    st.error(f"Application error: {str(e)}")