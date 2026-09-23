 import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 1. UI SETUP ---
st.set_page_config(page_title="Liquidity Sweep Backtester", layout="wide")
st.title("XAUT/USD Liquidity Sweep V2 - Backtesting App")

# Sidebar for Strategy Parameters
st.sidebar.header("Strategy Parameters")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=1000)
lot_size = st.sidebar.number_input("Lot Size (oz)", value=1.0) 
pivot_len = st.sidebar.number_input("Liquidity Swing", min_value=2, value=3)
confirm_bars = st.sidebar.number_input("MSS Confirmation Bars", min_value=1, value=6)
ema_len = st.sidebar.number_input("H1 EMA Length", min_value=10, value=50)
atr_len = st.sidebar.number_input("ATR Length", min_value=5, value=14)
rr_ratio = st.sidebar.slider("Risk Reward", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
use_trend = st.sidebar.checkbox("Use Trend Filter", value=True)
days_history = st.sidebar.slider("Days of Data (15m TF)", min_value=5, max_value=59, value=30)

# --- 2. DATA FETCHING ---
@st.cache_data(ttl=900)
def get_data(days):
    df = yf.download("GC=F", period=f"{days}d", interval="15m")
    df_h1 = yf.download("GC=F", period="60d", interval="1h")
    
    # Calculate H1 EMA manually
    df_h1['H1_EMA'] = df_h1['Close'].ewm(span=ema_len, adjust=False).mean()
    df_h1 = df_h1[['H1_EMA']].resample('15T').ffill()
    
    df = df.join(df_h1, how='left').ffill()
    df.dropna(inplace=True)
    return df

with st.spinner("Fetching data and running backtest..."):
    df = get_data(days_history)

# --- 3. INDICATORS & CONDITIONS ---
if not df.empty:
    df['Bull_Trend'] = df['Close'] > df['H1_EMA']
    df['Bear_Trend'] = df['Close'] < df['H1_EMA']

    df['Pivot_High'] = df['High'].rolling(window=pivot_len*2+1, center=True).max()
    df['Pivot_Low'] = df['Low'].rolling(window=pivot_len*2+1, center=True).min()
    df['Liquidity_High'] = df['Pivot_High'].ffill()
    df['Liquidity_Low'] = df['Pivot_Low'].ffill()

    df['Bull_Sweep'] = (df['Low'] < df['Liquidity_Low']) & (df['Close'] > df['Liquidity_Low'])
    df['Bear_Sweep'] = (df['High'] > df['Liquidity_High']) & (df['Close'] < df['Liquidity_High'])

    df['Struct_High'] = df['High'].shift(1).rolling(pivot_len * 2).max()
    df['Struct_Low'] = df['Low'].shift(1).rolling(pivot_len * 2).min()
    
    # Calculate ATR manually (TradingView RMA style)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/atr_len, adjust=False).mean()

    # --- 4. BACKTESTING ENGINE ---
    balance = initial_capital
    equity_curve = []
    trade_log = []
    
    in_pos = False
    pos_type = None
    entry_price, sl, tp = 0, 0, 0
    
    active_long_setup = 0
    active_short_setup = 0
    current_sweep_low = np.nan
    current_sweep_high = np.nan

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        
        # Manage Active Positions (Exits)
        if in_pos:
            if pos_type == 'LONG':
                if row['Low'] <= sl: 
                    loss = (sl - entry_price) * lot_size
                    balance += loss
                    trade_log.append({'Date': date, 'Type': 'LONG', 'Entry': entry_price, 'Exit': sl, 'Result': 'Loss', 'PnL': loss, 'Balance': balance})
                    in_pos = False
                elif row['High'] >= tp: 
                    profit = (tp - entry_price) * lot_size
                    balance += profit
                    trade_log.append({'Date': date, 'Type': 'LONG', 'Entry': entry_price, 'Exit': tp, 'Result': 'Win', 'PnL': profit, 'Balance': balance})
                    in_pos = False
            
            elif pos_type == 'SHORT':
                if row['High'] >= sl: 
                    loss = (entry_price - sl) * lot_size
                    balance += loss
                    trade_log.append({'Date': date, 'Type': 'SHORT', 'Entry': entry_price, 'Exit': sl, 'Result': 'Loss', 'PnL': loss, 'Balance': balance})
                    in_pos = False
                elif row['Low'] <= tp: 
                    profit = (entry_price - tp) * lot_size
                    balance += profit
                    trade_log.append({'Date': date, 'Type': 'SHORT', 'Entry': entry_price, 'Exit': tp, 'Result': 'Win', 'PnL': profit, 'Balance': balance})
                    in_pos = False

        equity_curve.append(balance)

        # Manage Setups
        if active_long_setup > 0: active_long_setup -= 1
        if active_short_setup > 0: active_short_setup -= 1

        if row['Bull_Sweep']:
            active_long_setup = confirm_bars
            current_sweep_low = row['Low']
        if row['Bear_Sweep']:
            active_short_setup = confirm_bars
            current_sweep_high = row['High']

        bull_mss = (active_long_setup > 0) and (row['Close'] > row['Struct_High'])
        bear_mss = (active_short_setup > 0) and (row['Close'] < row['Struct_Low'])

        # Enter New Positions
        if not in_pos:
            if bull_mss and (not use_trend or row['Bull_Trend']):
                sl = current_sweep_low - (row['ATR'] * 0.15)
                risk = row['Close'] - sl
                if risk > 0:
                    tp = row['Close'] + (risk * rr_ratio)
                    entry_price = row['Close']
                    pos_type = 'LONG'
                    in_pos = True
                    active_long_setup = 0 
            
            elif bear_mss and (not use_trend or row['Bear_Trend']):
                sl = current_sweep_high + (row['ATR'] * 0.15)
                risk = sl - row['Close']
                if risk > 0:
                    tp = row['Close'] - (risk * rr_ratio)
                    entry_price = row['Close']
                    pos_type = 'SHORT'
                    in_pos = True
                    active_short_setup = 0 

    df['Equity'] = equity_curve
    trade_df = pd.DataFrame(trade_log)

    # --- 5. UI DISPLAY ---
    tab1, tab2, tab3 = st.tabs(["Dashboard & Metrics", "Equity Curve", "Trade Log"])

    with tab1:
        total_trades = len(trade_df)
        if total_trades > 0:
            winning_trades = len(trade_df[trade_df['Result'] == 'Win'])
            win_rate = (winning_trades / total_trades) * 100
            net_profit = balance - initial_capital
        else:
            win_rate = 0
            net_profit = 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Balance", f"${balance:,.2f}", f"{net_profit:,.2f}")
        col2.metric("Total Trades", total_trades)
        col3.metric("Win Rate", f"{win_rate:.1f}%")
        col4.metric("Risk / Reward", f"1 : {rr_ratio}")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Liquidity_High'], mode='lines', line=dict(color='red', width=1, dash='dot'), name='Liq High'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Liquidity_Low'], mode='lines', line=dict(color='green', width=1, dash='dot'), name='Liq Low'))
        
        if not trade_df.empty:
            longs = trade_df[trade_df['Type'] == 'LONG']
            shorts = trade_df[trade_df['Type'] == 'SHORT']
            fig.add_trace(go.Scatter(x=longs['Date'], y=longs['Entry'] - 5, mode='markers', marker=dict(color='blue', size=10, symbol='triangle-up'), name='Long Entry'))
            fig.add_trace(go.Scatter(x=shorts['Date'], y=shorts['Entry'] + 5, mode='markers', marker=dict(color='magenta', size=10, symbol='triangle-down'), name='Short Entry'))

        fig.update_layout(height=600, template='plotly_dark', title="Price Action & Trade Entries", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=df.index, y=df['Equity'], mode='lines', line=dict(color='#00ff00', width=2), name='Equity', fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)'))
        fig_equity.update_layout(height=500, template='plotly_dark', title="Account Equity Growth")
        st.plotly_chart(fig_equity, use_container_width=True)

    with tab3:
        if not trade_df.empty:
            st.dataframe(trade_df.style.map(lambda x: 'color: green' if x == 'Win' else ('color: red' if x == 'Loss' else ''), subset=['Result']))
        else:
            st.info("No trades executed during this period with the current parameters.")

else:
    st.error("Failed to fetch data.")
