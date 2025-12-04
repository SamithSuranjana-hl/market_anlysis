import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import os

# Check for alpaca-py installation
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Page configuration
st.set_page_config(page_title="Trading Strategy Analyzer", layout="wide")

# Title
st.title("📈 Advanced Trading Strategy Analyzer")
st.markdown("---")

# Sidebar controls
st.sidebar.header("Strategy Parameters")

# Data source selection
data_source = st.sidebar.radio(
    "Data Source",
    ["Load Local File", "Download from Alpaca"] if ALPACA_AVAILABLE else ["Load Local File"],
    index=0
)

if data_source == "Load Local File":
    uploaded_file = st.sidebar.file_uploader("Upload Parquet File", type=['parquet'])
    symbol = st.sidebar.text_input("Symbol", "META")
else:
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Symbol",
        ["META", "AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "SPY", "QQQ"],
        index=0
    )
    # Time period
    years = st.sidebar.slider("Years of Historical Data", 1, 5, 3)

# Strategy parameters
st.sidebar.subheader("Risk Management")
initial_equity = st.sidebar.number_input("Initial Equity ($)", 1000, 1000000, 10000, 1000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1) / 100
risk_reward = st.sidebar.slider("Risk/Reward Ratio", 1.0, 5.0, 2.0, 0.5)
slippage = st.sidebar.slider("Slippage (%)", 0.0, 1.0, 0.0, 0.01) / 100

st.sidebar.subheader("Technical Indicators")
ema_period = st.sidebar.slider("EMA Period (Days)", 20, 200, 50, 10)
cvd_ema_span = st.sidebar.slider("CVD EMA Span", 10, 50, 25, 5)
cvd_mom_window = st.sidebar.slider("CVD Momentum Window", 3, 20, 7, 1)

# Stop loss / Take profit percentages
sl_pct = st.sidebar.slider("Stop Loss (%)", 0.1, 2.0, 0.5, 0.1) / 100
tp_pct = sl_pct * risk_reward

# Data action button
if data_source == "Download from Alpaca":
    download_data = st.sidebar.button("📥 Download Fresh Data", type="primary")
else:
    download_data = False
    load_data = st.sidebar.button("📊 Load Data", type="primary") if uploaded_file else False

@st.cache_data
def download_market_data(symbol, years):
    """Download historical data from Alpaca"""
    if not ALPACA_AVAILABLE:
        st.error("❌ alpaca-py is not installed. Install it with: pip install alpaca-py")
        return None
    
    try:
        API_KEY = "PKJKC7W6ISEM5TDKIIX3HKQE64"
        SECRET_KEY = "3WnsZUNSe4d2VJErpbFwjBoB4awuG3gSEBkE8ycDck4i"
        
        if not API_KEY or not SECRET_KEY:
            st.error("❌ Alpaca API credentials not found in environment variables")
            st.info("Set them with: export ALPACA_API_KEY='your_key' and export ALPACA_SECRET_KEY='your_secret'")
            return None
        
        client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=365 * years)
        
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=start_date,
            end=end_date,
            feed="iex",
            adjustment="all"
        )
        
        bars = client.get_stock_bars(request).df
        bars = bars.reset_index()
        
        return bars
    except Exception as e:
        st.error(f"❌ Error downloading data: {str(e)}")
        return None

def load_local_data(uploaded_file):
    """Load data from uploaded parquet file"""
    try:
        df = pd.read_parquet(uploaded_file)
        return df
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def prepare_data(df, symbol):
    """Prepare and clean the data"""
    df = df[df["symbol"] == symbol].drop(columns=["symbol"])
    df = df.rename(columns={
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })
    df.set_index("Date", inplace=True)
    return df

def add_indicators(df, ema_period, cvd_ema_span, cvd_mom_window):
    """Add technical indicators"""
    # VWAP
    df['cum_vol_price'] = (df['Volume'] * df['Close']).cumsum()
    df['cum_vol'] = df['Volume'].cumsum()
    df['VWAP'] = df['cum_vol_price'] / df['cum_vol']
    
    # CVD
    df['delta'] = np.where(
        df['Close'] > df['Open'], df['Volume'],
        np.where(df['Close'] < df['Open'], -df['Volume'], 0)
    )
    df['CVD'] = df['delta'].cumsum()
    df['CVD_sm'] = df['CVD'].ewm(span=cvd_ema_span, adjust=False).mean()
    df['cvd_mom'] = df['CVD_sm'] - df['CVD_sm'].shift(1)
    df['cvd_mean'] = df['cvd_mom'].rolling(cvd_mom_window).mean()
    
    # Daily EMA Regime
    daily = df['Close'].resample('1D').last().ffill()
    ema = daily.ewm(span=ema_period, adjust=False).mean()
    df['daily_close'] = df['Close'].resample('1D').ffill().reindex(df.index, method='ffill')
    df['daily_EMA'] = ema.reindex(df.index, method='ffill')
    df['regime_long'] = df['daily_close'] > df['daily_EMA']
    df['regime_short'] = df['daily_close'] < df['daily_EMA']
    
    return df

def generate_signals(df):
    """Generate trading signals"""
    df['long_signal'] = (
        (df['Close'] > df['VWAP']) &
        (df['cvd_mom'] > df['cvd_mean']) &
        (df['regime_long'])
    ).astype(int)
    
    df['short_signal'] = (
        (df['Close'] < df['VWAP']) &
        (df['cvd_mom'] < df['cvd_mean']) &
        (df['regime_short'])
    ).astype(int)
    
    return df

def intrabar_exit(row, sl_price, tp_price, direction):
    """Determine exit within the bar"""
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
    
    if direction == "long":
        if c > o:
            if l <= sl_price: return sl_price, "SL"
            if h >= tp_price: return tp_price, "TP"
        elif c < o:
            if h >= tp_price: return tp_price, "TP"
            if l <= sl_price: return sl_price, "SL"
        else:
            if l <= sl_price: return sl_price, "SL"
            if h >= tp_price: return tp_price, "TP"
    
    if direction == "short":
        if c > o:
            if h >= sl_price: return sl_price, "SL"
            if l <= tp_price: return tp_price, "TP"
        elif c < o:
            if l <= tp_price: return tp_price, "TP"
            if h >= sl_price: return sl_price, "SL"
        else:
            if h >= sl_price: return sl_price, "SL"
            if l <= tp_price: return tp_price, "TP"
    
    return None, None

def backtest(df, initial_equity, risk_per_trade, sl_pct, tp_pct, slippage):
    """Run the backtest"""
    equity = initial_equity
    pos = None
    trades = []
    equity_curve = [initial_equity]
    trade_log = []
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Check active trade
        if pos is not None:
            exit_price, exit_reason = intrabar_exit(
                row, pos['sl_price'], pos['tp_price'], pos['direction']
            )
            
            if exit_price is not None:
                exit_price *= (1 - slippage)
                
                if pos['direction'] == "long":
                    trade_ret = (exit_price - pos['entry_price']) / pos['entry_price']
                else:
                    trade_ret = (pos['entry_price'] - exit_price) / pos['entry_price']
                
                equity *= (1 + trade_ret)
                trades.append(trade_ret)
                
                trade_log.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': row.name,
                    'direction': pos['direction'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'return': trade_ret,
                    'exit_reason': exit_reason
                })
                
                pos = None
        
        equity_curve.append(equity)
        
        # New entry
        if pos is None:
            if prev['long_signal'] == 1:
                entry_price = row['Open']
                dollar_risk = equity * risk_per_trade
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)
                
                pos = {
                    "direction": "long",
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "entry_time": row.name
                }
            
            elif prev['short_signal'] == 1:
                entry_price = row['Open']
                dollar_risk = equity * risk_per_trade
                sl_price = entry_price * (1 + sl_pct)
                tp_price = entry_price * (1 - tp_pct)
                
                pos = {
                    "direction": "short",
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "entry_time": row.name
                }
    
    return trades, equity_curve, trade_log

# Main execution
if data_source == "Download from Alpaca" and (download_data or 'df' not in st.session_state):
    with st.spinner(f"Downloading {symbol} data..."):
        raw_data = download_market_data(symbol, years)
        if raw_data is not None:
            st.session_state['raw_data'] = raw_data
            st.session_state['symbol'] = symbol
            st.success(f"✅ Downloaded {len(raw_data):,} bars")
elif data_source == "Load Local File" and uploaded_file is not None:
    with st.spinner("Loading data from file..."):
        raw_data = load_local_data(uploaded_file)
        if raw_data is not None:
            st.session_state['raw_data'] = raw_data
            st.session_state['symbol'] = symbol
            st.success(f"✅ Loaded {len(raw_data):,} bars")

if 'raw_data' in st.session_state:
    # Prepare data
    with st.spinner("Processing data..."):
        df = prepare_data(st.session_state['raw_data'].copy(), st.session_state['symbol'])
        df = add_indicators(df, ema_period, cvd_ema_span, cvd_mom_window)
        df = generate_signals(df)
        st.session_state['df'] = df
    
    # Run backtest
    with st.spinner("Running backtest..."):
        trades, equity_curve, trade_log = backtest(
            df, initial_equity, risk_per_trade, sl_pct, tp_pct, slippage
        )
    
    # Calculate metrics
    if len(trades) > 0:
        total_return = (equity_curve[-1] - initial_equity) / initial_equity * 100
        win_rate = sum(1 for t in trades if t > 0) / len(trades) * 100
        avg_win = np.mean([t for t in trades if t > 0]) * 100 if any(t > 0 for t in trades) else 0
        avg_loss = np.mean([t for t in trades if t < 0]) * 100 if any(t < 0 for t in trades) else 0
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Return", f"{total_return:.2f}%")
        col2.metric("Total Trades", len(trades))
        col3.metric("Win Rate", f"{win_rate:.1f}%")
        col4.metric("Avg Win", f"{avg_win:.2f}%")
        col5.metric("Avg Loss", f"{avg_loss:.2f}%")
        
        # Equity curve chart
        st.subheader("📊 Equity Curve")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=df.index[:len(equity_curve)],
            y=equity_curve,
            mode='lines',
            name='Equity',
            line=dict(color='#00cc96', width=2)
        ))
        fig_equity.update_layout(
            height=400,
            hovermode='x unified',
            xaxis_title="Date",
            yaxis_title="Equity ($)"
        )
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Price chart with signals
        st.subheader("💹 Price Action & Signals")
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxis=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price & VWAP', 'CVD Momentum', 'Volume')
        )
        
        # Sample data for visualization (last 1000 bars)
        plot_df = df.iloc[-1000:]
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df['Open'],
            high=plot_df['High'],
            low=plot_df['Low'],
            close=plot_df['Close'],
            name='Price'
        ), row=1, col=1)
        
        # VWAP
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df['VWAP'],
            mode='lines',
            name='VWAP',
            line=dict(color='orange', width=2)
        ), row=1, col=1)
        
        # Long signals
        long_sigs = plot_df[plot_df['long_signal'] == 1]
        fig.add_trace(go.Scatter(
            x=long_sigs.index,
            y=long_sigs['Low'] * 0.998,
            mode='markers',
            name='Long Signal',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ), row=1, col=1)
        
        # Short signals
        short_sigs = plot_df[plot_df['short_signal'] == 1]
        fig.add_trace(go.Scatter(
            x=short_sigs.index,
            y=short_sigs['High'] * 1.002,
            mode='markers',
            name='Short Signal',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ), row=1, col=1)
        
        # CVD Momentum
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df['cvd_mom'],
            mode='lines',
            name='CVD Mom',
            line=dict(color='blue')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df['cvd_mean'],
            mode='lines',
            name='CVD Mean',
            line=dict(color='red', dash='dash')
        ), row=2, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=plot_df.index,
            y=plot_df['Volume'],
            name='Volume',
            marker_color='lightblue'
        ), row=3, col=1)
        
        fig.update_layout(height=900, hovermode='x unified', showlegend=True)
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade log
        st.subheader("📋 Recent Trades")
        if trade_log:
            trade_df = pd.DataFrame(trade_log[-20:])
            trade_df['return'] = trade_df['return'] * 100
            st.dataframe(
                trade_df.style.format({
                    'entry_price': '${:.2f}',
                    'exit_price': '${:.2f}',
                    'return': '{:.2f}%'
                }).background_gradient(subset=['return'], cmap='RdYlGn', vmin=-2, vmax=2),
                use_container_width=True
            )
        else:
            st.info("No trades executed yet")
    else:
        st.warning("⚠️ No trades were executed with current parameters. Try adjusting the strategy settings.")
else:
    st.info("👆 Upload a parquet file or download data from Alpaca to start the analysis")
    
    # Show installation instructions if Alpaca is not available
    if not ALPACA_AVAILABLE:
        st.warning("⚠️ alpaca-py is not installed")
        st.code("pip install alpaca-py", language="bash")
        st.markdown("After installation, you'll be able to download live data from Alpaca Markets.")
    
    # Example of expected data format
    with st.expander("ℹ️ Expected Data Format"):
        st.markdown("""
        Your parquet file should have the following columns:
        - `symbol`: Stock symbol (e.g., 'META')
        - `timestamp`: Datetime with timezone
        - `open`: Opening price
        - `high`: High price
        - `low`: Low price
        - `close`: Closing price
        - `volume`: Volume
        - `trade_count`: Number of trades (optional)
        - `vwap`: VWAP (optional)
        """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit • Data powered by Alpaca Markets")
