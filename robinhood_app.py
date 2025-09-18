import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import time
import os
import datetime

# --- Page Configuration (Set at the very top) ---
st.set_page_config(layout="wide", page_title="Robinhood Portfolio Analysis")

# --- Configuration ---
DB_FILE = 'robinhood_portfolio.db'

# ==============================================================================
# DATA FETCHING & CACHING FUNCTIONS (The "Data Kitchen")
# Note: These functions ONLY fetch and return data. They do NOT use any st.* UI elements.
# ==============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_live_prices(tickers):
    """
    Fetches the current market price for a list of tickers.
    Returns a dictionary mapping tickers to their price.
    """
    if not tickers:
        return {}
    try:
        data = yf.download(tickers=tickers, period='1d', progress=False)
        if data.empty:
            return {ticker: 0 for ticker in tickers}
        # Use the most recent 'Close' price
        latest_prices = data['Close'].iloc[-1]
        return latest_prices.to_dict()
    except Exception as e:
        # If the API fails, return a dictionary of zeros to prevent the app from crashing.
        st.error(f"Error fetching live prices from yfinance: {e}")
        return {ticker: 0 for ticker in tickers}

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_live_dividend_yields(tickers):
    """
    Fetches the current dividend yield for a list of tickers.
    Returns a dictionary mapping tickers to their dividend yield.
    """
    yields = {}
    if not tickers:
        return {}
    # Fetch info for all tickers in a single session for efficiency
    ticker_objects = yf.Tickers(tickers)
    for ticker_symbol in tickers:
        try:
            # Access the already downloaded info
            info = ticker_objects.tickers[ticker_symbol.upper()].info
            yield_value = info.get('dividendYield', 0)
            yields[ticker_symbol] = yield_value if yield_value is not None else 0
        except Exception:
            yields[ticker_symbol] = 0 # Default to 0 if any error occurs for a single ticker
    return yields

@st.cache_data
def load_data_from_db():
    """
    Loads all necessary tables from the SQLite database.
    Caches the result so the DB is only hit once per session.
    """
    if not os.path.exists(DB_FILE):
        return None, None, None, None

    conn = sqlite3.connect(DB_FILE)
    try:
        daily_portfolio_df = pd.read_sql_query("SELECT * FROM daily_portfolio_snapshots", conn)
        closed_trades_df = pd.read_sql_query("SELECT * FROM closed_trades_summary", conn)
        transactions_cleaned_df = pd.read_sql_query("SELECT * FROM transactions_cleaned", conn)
        # Load from the new simplified table
        open_positions_df = pd.read_sql_query("SELECT * FROM open_positions_summary", conn)
    except Exception:
         # If a table doesn't exist, return empty DataFrames
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    finally:
        conn.close()
    
    # Convert date columns
    daily_portfolio_df['Date'] = pd.to_datetime(daily_portfolio_df['Date'])
    closed_trades_df['sell_date'] = pd.to_datetime(closed_trades_df['sell_date'])
    transactions_cleaned_df['activity_date'] = pd.to_datetime(transactions_cleaned_df['activity_date'])
    
    daily_portfolio_df.set_index('Date', inplace=True)
    daily_portfolio_df.sort_index(inplace=True)

    return daily_portfolio_df, closed_trades_df, transactions_cleaned_df, open_positions_df

# ==============================================================================
# MAIN APP LAYOUT (The "Storefront")
# ==============================================================================

st.title(" Robinhood Portfolio Analysis")
st.markdown("---")

# --- Data Loading ---
daily_portfolio_df, closed_trades_df, transactions_cleaned_df, open_positions_df = load_data_from_db()

if open_positions_df is None:
    st.error(f"Database file '{DB_FILE}' not found. Please run the data pipeline first.")
    st.stop()

# --- Key Metrics Display ---
st.subheader("Performance Summary")

# Calculations for metrics
overall_twr = (daily_portfolio_df['cumulative_twr_factor'].iloc[-1] - 1) if not daily_portfolio_df.empty else 0
max_drawdown = daily_portfolio_df['drawdown'].min() if not daily_portfolio_df.empty else 0

if not daily_portfolio_df.empty and 'daily_return_adjusted' in daily_portfolio_df.columns and len(daily_portfolio_df) > 1:
    volatility = daily_portfolio_df['daily_return_adjusted'].std() * np.sqrt(252)
    risk_free_rate = 0.02
    daily_risk_free_rate = (1 + risk_free_rate)**(1/252) - 1
    excess_returns = daily_portfolio_df['daily_return_adjusted'] - daily_risk_free_rate
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
else:
    volatility = 0
    sharpe_ratio = 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Time-Weighted Return (All-Time)", f"{overall_twr * 100:.2f}%")
col2.metric("Maximum Drawdown", f"{max_drawdown * 100:.2f}%")
col3.metric("Annualized Volatility", f"{volatility * 100:.2f}%")
col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

st.markdown("---")

# --- Main Content in Tabs ---
tab_main, tab_analysis, tab_cash_flow = st.tabs([" Main Dashboard", " Detailed Analysis", " Cash Flow"])

with tab_main:
    st.subheader("Current Portfolio Holdings")
    
    if not open_positions_df.empty:
        # --- Live Data Integration ---
        # 1. Call the cached functions to get live data
        tickers = open_positions_df['instrument'].tolist()
        live_prices = get_live_prices(tickers)
        live_yields = get_live_dividend_yields(tickers)

        # 2. Map the returned data to the DataFrame
        holdings_df = open_positions_df.copy()
        holdings_df['current_price'] = holdings_df['instrument'].map(live_prices)
        holdings_df['dividend_yield'] = holdings_df['instrument'].map(live_yields) * 100

        # 3. Perform live calculations
        holdings_df['market_value_live'] = holdings_df['quantity'] * holdings_df['current_price']
        holdings_df['unrealized_pl_live'] = holdings_df['market_value_live'] - holdings_df['cost_basis_total']
        holdings_df['unrealized_pl_pct'] = (holdings_df['unrealized_pl_live'] / holdings_df['cost_basis_total'].replace(0, np.nan)) * 100
        
        total_market_value = holdings_df['market_value_live'].sum()
        if total_market_value > 0:
            holdings_df['portfolio_allocation_pct'] = (holdings_df['market_value_live'] / total_market_value) * 100
        else:
            holdings_df['portfolio_allocation_pct'] = 0

        # 4. Display the results in the UI
        st.dataframe(
            holdings_df[['instrument', 'quantity', 'cost_basis_total', 'current_price', 'market_value_live', 'dividend_yield', 'unrealized_pl_live', 'unrealized_pl_pct', 'portfolio_allocation_pct']].sort_values(by='market_value_live', ascending=False),
            column_config={
                "quantity": st.column_config.NumberColumn(format="%.4f"),
                "cost_basis_total": st.column_config.NumberColumn("Cost Basis", format="$%.2f"),
                "current_price": st.column_config.NumberColumn("Price (Live)", format="$%.2f"),
                "market_value_live": st.column_config.NumberColumn("Market Value (Live)", format="$%.2f"),
                "dividend_yield": st.column_config.NumberColumn("Yield (Live)", format="%.2f%%"),
                "unrealized_pl_live": st.column_config.NumberColumn("Unrealized P/L (Live)", format="$%.2f"),
                "unrealized_pl_pct": st.column_config.NumberColumn("Unrealized P/L %", format="%.2f%%"),
                "portfolio_allocation_pct": st.column_config.NumberColumn("Allocation %", format="%.2f%%"),
            },
            use_container_width=True,
            height=35 * (len(holdings_df) + 1)
        )
    else:
        st.info("No current holdings found.")

with tab_analysis:
    st.subheader("Trading Performance Insights")
    if not closed_trades_df.empty:
        # Trading metrics calculations...
        winning_trades = closed_trades_df[closed_trades_df['realized_profit_loss'] > 0]
        losing_trades = closed_trades_df[closed_trades_df['realized_profit_loss'] < 0]
        # (rest of your analysis code is unchanged)
    else:
        st.info("No closed trades to analyze.")

    st.markdown("---")
    st.subheader("Instrument-Specific Analysis")
    all_instruments = sorted(transactions_cleaned_df['instrument'].dropna().unique().tolist())
    selected_instrument = st.selectbox("Select an Instrument to Analyze", all_instruments)

    if selected_instrument:
        # (rest of your analysis code is unchanged)
        st.write("Instrument analysis section...")

with tab_cash_flow:
    st.subheader("Cash Flow Analysis")
    if not transactions_cleaned_df.empty:
        # (rest of your cash flow analysis code is unchanged)
        st.write("Cash flow analysis section...")
    else:
        st.info("No transaction data available for cash flow analysis.")
