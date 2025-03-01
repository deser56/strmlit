import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import websocket
import json
import threading
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from scipy.signal import argrelextrema
import ta

st.set_page_config(page_title="Quantum Trader Pro", layout="wide", page_icon="🚀")

if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None

@st.cache_data
def get_multi_tf_data(ticker, primary_tf, secondary_tf, lookback):
    def fetch_data(interval):
        try:
            data = yf.Ticker(ticker).history(period=f"{lookback}d", interval=interval)
            return data if not data.empty else pd.DataFrame()
        except Exception as e:
            st.error(f"Data error: {str(e)}")
            return pd.DataFrame()
    
    return fetch_data(primary_tf), fetch_data(secondary_tf)

with st.sidebar:
    st.header("Quantum Configuration ⚛️")
    ticker = st.text_input("Asset Symbol", "AAPL").upper()
    primary_tf = st.selectbox("Primary Timeframe", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"])
    secondary_tf = st.selectbox("Secondary Timeframe", ["5m", "15m", "30m", "1h", "1d"])
    lookback = st.slider("Analysis Window (Days)", 1, 7, 7)
    enable_ml = st.checkbox("Enable ML Predictions", True)
    enable_ichimoku = st.checkbox("Ichimoku Cloud", True)
    enable_fib = st.checkbox("Fibonacci Retracement", True)
    risk_per_trade = st.number_input("Risk % per Trade", 0.1, 5.0, 1.0)

def calculate_advanced_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()

    if enable_ichimoku:
        ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
        df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_SpanA'] = ichimoku.ichimoku_a()
        df['Ichimoku_SpanB'] = ichimoku.ichimoku_b()

    if enable_fib:
        window = min(50, len(df))
        recent_high = df['High'].rolling(window, min_periods=1).max().iloc[-1]
        recent_low = df['Low'].rolling(window, min_periods=1).min().iloc[-1]
        diff = recent_high - recent_low
        for level, val in zip([0.236, 0.382, 0.5, 0.618, 0.786], 
                            [recent_high - diff * x for x in [0.236, 0.382, 0.5, 0.618, 0.786]]):
            df[f'Fib_{level}'] = val

    return df

def create_ml_model(df):
    features = [f for f in ['RSI', 'MACD', 'VWAP', 'ATR'] if f in df.columns]
    if not features: return None, None
    
    model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('regressor', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    X = df[features].dropna()
    y = df['Close'].shift(-3).dropna()
    common_idx = X.index.intersection(y.index)
    
    if len(common_idx) > 100:
        model.fit(X.loc[common_idx], y.loc[common_idx])
        return model, features
    return None, None

def detect_chart_patterns(df):
    patterns = []
    try:
        max_idx = argrelextrema(df['Close'].values, np.greater, order=5)[0]
        if len(max_idx) > 3 and (df['Close'].iloc[max_idx[-3]] > df['Close'].iloc[max_idx[-4]] and 
                                df['Close'].iloc[max_idx[-3]] > df['Close'].iloc[max_idx[-2]]):
            patterns.append('Head & Shoulders')
    except: pass
    return patterns

def calculate_position_size(price, stop_loss):
    risk_per_share = abs(price - stop_loss)
    return 0 if risk_per_share < 1e-6 else round((100000 * (risk_per_trade/100)) / risk_per_share)

def create_advanced_chart(primary_df, secondary_df):
    fig = make_subplots(rows=4, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.5, 0.2, 0.2, 0.1],
                       specs=[[{"secondary_y": True}],
                              [{"secondary_y": False}],
                              [{"secondary_y": False}],
                              [{"secondary_y": False}]])
    
    # Primary Price Chart
    fig.add_trace(go.Candlestick(x=primary_df.index,
                                open=primary_df['Open'],
                                high=primary_df['High'],
                                low=primary_df['Low'],
                                close=primary_df['Close'],
                                name='Price'), row=1, col=1)
    
    # ML Predictions
    if enable_ml and 'ml_model' in st.session_state and 'ml_features' in st.session_state:
        model = st.session_state.ml_model
        features = st.session_state.ml_features
        if model and features:
            X_pred = primary_df[features].dropna()
            if not X_pred.empty:
                predictions = model.predict(X_pred)
                fig.add_trace(go.Scatter(x=X_pred.index, y=predictions, 
                                       line=dict(color='gold'), name='ML Forecast'),
                            row=1, col=1)

    # Ichimoku Cloud
    if enable_ichimoku:
        fig.add_trace(go.Scatter(x=primary_df.index, 
                              y=primary_df['Ichimoku_SpanA'],
                              fill='tonexty',
                              line=dict(color='rgba(0,150,255,0.2)'),
                              name='Ichimoku Cloud'), row=1, col=1)
        fig.add_trace(go.Scatter(x=primary_df.index, 
                              y=primary_df['Ichimoku_SpanB'],
                              fill='tonexty',
                              line=dict(color='rgba(255,100,0,0.2)'),
                              name='Span B'), row=1, col=1)

    # Volume Chart
    fig.add_trace(go.Bar(x=secondary_df.index,
                       y=secondary_df['Volume'],
                       marker_color='rgba(100,200,255,0.6)'), row=4, col=1)

    fig.update_layout(height=1000,
                     xaxis_rangeslider_visible=False,
                     template='plotly_dark')
    return fig

class QuantumWebSocket:
    def __init__(self, ticker):
        self.ws = websocket.WebSocketApp("wss://example.com/placeholder",
                                        on_message=lambda ws, msg: self.on_message(ws, msg),
                                        on_error=lambda ws, err: st.error(f"WS Error: {err}"),
                                        on_close=lambda ws: st.warning("WS Closed"))
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()

try:
    st.markdown("## Quantum Trading Suite Pro 🌌")
    primary_df, secondary_df = get_multi_tf_data(ticker, primary_tf, secondary_tf, lookback)
    
    if not primary_df.empty:
        primary_df = calculate_advanced_indicators(primary_df)
        if st.session_state.current_ticker != ticker:
            if 'ws' in st.session_state: del st.session_state.ws
            st.session_state.current_ticker = ticker
        if 'ws' not in st.session_state:
            st.session_state.ws = QuantumWebSocket(ticker)

        if enable_ml:
            model, features = create_ml_model(primary_df)
            if model:
                st.session_state.ml_model = model
                st.session_state.ml_features = features

        st.plotly_chart(create_advanced_chart(primary_df, secondary_df), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Market Psychology 🧠")
            st.metric("RSI", f"{primary_df['RSI'].iloc[-1]:.1f}")

        with col2:
            st.header("Risk Matrix ⚠️")
            atr = primary_df['ATR'].iloc[-1]
            st.metric("Position Size", calculate_position_size(primary_df['Close'].iloc[-1], 
                      primary_df['Close'].iloc[-1] - atr))

        with col3:
            st.header("Patterns")
            for pattern in detect_chart_patterns(primary_df):
                st.success(pattern)

except Exception as e:
    st.error(f"System Error: {str(e)}")