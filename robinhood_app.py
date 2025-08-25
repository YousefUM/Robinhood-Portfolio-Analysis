import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from collections import deque
import numpy as np

# --- Configuration ---
DB_FILE = 'robinhood_portfolio.db'

# --- Custom Functions for Calculations (re-implemented from our V2 pipeline) ---
@st.cache_data
def get_current_holdings(transactions_df):
    """
    Calculates the current open positions and their cost basis using FIFO logic.
    """
    current_open_positions = {}
    instrument_relevant_actions = transactions_df[
        transactions_df['transaction_category'] == 'Trade'
    ].copy()
    if instrument_relevant_actions.empty:
        return pd.DataFrame(columns=['instrument', 'quantity', 'cost_basis_total'])
    grouped_by_instrument = instrument_relevant_actions.groupby('instrument')
    for instrument_name, group in grouped_by_instrument:
        buy_lots_for_instrument = deque()
        for index, row in group.iterrows():
            trans_code = row['trans_code']
            quantity = row['quantity']
            price = row['price']
            activity_date = row['activity_date']
            if trans_code == 'Buy':
                if pd.notna(quantity) and quantity > 0 and pd.notna(price):
                    buy_lots_for_instrument.append({'quantity': quantity, 'price': price, 'date': activity_date})
            elif trans_code == 'Sell':
                if pd.notna(quantity) and quantity > 0:
                    sold_quantity_remaining = quantity
                    while sold_quantity_remaining > 1e-9 and buy_lots_for_instrument:
                        oldest_lot = buy_lots_for_instrument[0]
                        lot_quantity = oldest_lot['quantity']
                        quantity_to_sell_from_lot = min(sold_quantity_remaining, lot_quantity)
                        sold_quantity_remaining -= quantity_to_sell_from_lot
                        oldest_lot['quantity'] -= quantity_to_sell_from_lot
                        if oldest_lot['quantity'] < 1e-9:
                            buy_lots_for_instrument.popleft()
        if buy_lots_for_instrument:
            total_current_quantity = sum(lot['quantity'] for lot in buy_lots_for_instrument)
            total_current_cost_basis = sum(lot['quantity'] * lot['price'] for lot in buy_lots_for_instrument)
            if abs(total_current_quantity) > 1e-9:
                current_open_positions[instrument_name] = {
                    'quantity': total_current_quantity,
                    'cost_basis_total': total_current_cost_basis
                }
    final_holdings_list = [
        {'instrument': ticker, 'quantity': details['quantity'], 'cost_basis_total': details['cost_basis_total']}
        for ticker, details in current_open_positions.items()
    ]
    return pd.DataFrame(final_holdings_list)

# --- Step 1: Connect to the Database and Load Data ---
st.set_page_config(layout="wide")
st.header("Robinhood Portfolio Analysis")
st.markdown("---")

try:
    conn = sqlite3.connect(DB_FILE)
    daily_portfolio_df = pd.read_sql_query("SELECT * FROM daily_portfolio_snapshots", conn)
    daily_portfolio_df['Date'] = pd.to_datetime(daily_portfolio_df['Date'])
    closed_trades_df = pd.read_sql_query("SELECT * FROM closed_trades_summary", conn)
    transactions_cleaned_df = pd.read_sql_query("SELECT * FROM transactions_cleaned", conn)
    transactions_cleaned_df['activity_date'] = pd.to_datetime(transactions_cleaned_df['activity_date'])
    st.success(f"Successfully loaded data from '{DB_FILE}'")
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure '{DB_FILE}' is in the correct folder and the data pipeline has been run.")
    st.stop()
finally:
    if conn:
        conn.close()

daily_portfolio_df.set_index('Date', inplace=True)
daily_portfolio_df.sort_index(inplace=True)
closed_trades_df['sell_date'] = pd.to_datetime(closed_trades_df['sell_date'])

# --- Step 2: V2 Summary Metrics & Key Insights ---
st.subheader("Performance Summary")

overall_twr = (daily_portfolio_df['cumulative_twr_factor'].iloc[-1] - 1) * 100 if not daily_portfolio_df.empty else 0
max_drawdown = daily_portfolio_df['drawdown'].min() * 100 if not daily_portfolio_df.empty else 0

if not daily_portfolio_df.empty and 'daily_return_adjusted' in daily_portfolio_df.columns and len(daily_portfolio_df) > 1:
    volatility = daily_portfolio_df['daily_return_adjusted'].std() * np.sqrt(252) * 100
    risk_free_rate = 0.02
    daily_risk_free_rate = (1 + risk_free_rate)**(1/252) - 1
    excess_returns = daily_portfolio_df['daily_return_adjusted'] - daily_risk_free_rate
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
else:
    volatility = 0
    sharpe_ratio = 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Time-Weighted Return", f"{overall_twr:.2f}%")
col2.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
col3.metric("Annualized Volatility", f"{volatility:.2f}%")
col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

# --- Step 3: V2 Portfolio Value Visualization ---
st.subheader("Daily Portfolio Value Over Time")
fig_portfolio_value = px.line(daily_portfolio_df, y=['Total_Portfolio_Value', 'Cash_Balance', 'Stock_Market_Value'])
st.plotly_chart(fig_portfolio_value, use_container_width=True)

# (All other sections of your app: Benchmark, Drawdown, Realized P/L, Holdings... etc.)
# ...
# ...

# --- NEW SECTION: Trading Performance Insights ---
st.markdown("---")
st.subheader("Trading Performance Insights")

if not closed_trades_df.empty:
    winning_trades = closed_trades_df[closed_trades_df['realized_profit_loss'] > 0]
    losing_trades = closed_trades_df[closed_trades_df['realized_profit_loss'] < 0]

    total_trades = len(closed_trades_df)
    win_count = len(winning_trades)
    
    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
    
    # Safely calculate average gain and loss
    avg_win = winning_trades['realized_profit_loss'].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades['realized_profit_loss'].mean() if not losing_trades.empty else 0

    # Safely calculate Profit Factor
    total_gains = winning_trades['realized_profit_loss'].sum()
    total_losses = abs(losing_trades['realized_profit_loss'].sum())
    
    if total_losses > 0:
        profit_factor = total_gains / total_losses
    elif total_gains > 0:
        profit_factor = float('inf') # Only gains, no losses
    else:
        profit_factor = 0 # No gains or losses

    # Display metrics
    t_col1, t_col2, t_col3 = st.columns(3)
    t_col1.metric("Win Rate", f"{win_rate:.2f}%", help="The percentage of closed trades that were profitable.")
    t_col2.metric("Avg. Gain / Loss", f"${avg_win:,.2f} / ${abs(avg_loss):,.2f}", help="The average dollar amount for winning and losing trades.")
    t_col3.metric("Profit Factor", f"{profit_factor:.2f}", help="Total gains divided by total losses. A value > 1 indicates profitability.")

else:
    st.info("No closed trades found to calculate Win/Loss Ratio.")
