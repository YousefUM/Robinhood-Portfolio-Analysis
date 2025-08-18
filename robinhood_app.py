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
    This logic mirrors our Phase 7 realized P/L calculation but stores open lots.
    """
    current_open_positions = {}
    
    # We only care about Trade category for share quantity changes
    instrument_relevant_actions = transactions_df[
        transactions_df['transaction_category'] == 'Trade'
    ].copy()

    if instrument_relevant_actions.empty:
        return pd.DataFrame(columns=['instrument', 'quantity', 'cost_basis_total'])

    grouped_by_instrument = instrument_relevant_actions.groupby('instrument')

    for instrument_name, group in grouped_by_instrument:
        buy_lots_for_instrument = deque() # Stores {'quantity': float, 'price': float, 'date': datetime}
        
        for index, row in group.iterrows():
            trans_code = row['trans_code']
            quantity = row['quantity'] # This is the already adjusted quantity
            price = row['price']       # This is the already adjusted price
            activity_date = row['activity_date']

            if trans_code == 'Buy':
                if pd.notna(quantity) and quantity > 0 and pd.notna(price):
                     buy_lots_for_instrument.append({'quantity': quantity, 'price': price, 'date': activity_date})
                # Note: No fallback for missing price/amount here to keep it simple;
                # the pipeline should have already cleaned this up.

            elif trans_code == 'Sell':
                if pd.notna(quantity) and quantity > 0:
                    sold_quantity_remaining = quantity
                    while sold_quantity_remaining > 1e-9 and buy_lots_for_instrument:
                        oldest_lot = buy_lots_for_instrument[0]
                        lot_quantity = oldest_lot['quantity']
                        
                        quantity_to_sell_from_lot = min(sold_quantity_remaining, lot_quantity)
                        
                        sold_quantity_remaining -= quantity_to_sell_from_lot
                        oldest_lot['quantity'] -= quantity_to_sell_from_lot

                        if oldest_lot['quantity'] < 1e-9: # Lot fully consumed or negligible
                            buy_lots_for_instrument.popleft()
                            
        # After processing all transactions for this instrument, sum up remaining lots
        if buy_lots_for_instrument:
            total_current_quantity = sum(lot['quantity'] for lot in buy_lots_for_instrument)
            total_current_cost_basis = sum(lot['quantity'] * lot['price'] for lot in buy_lots_for_instrument)

            if abs(total_current_quantity) > 1e-9:
                 current_open_positions[instrument_name] = {
                    'quantity': total_current_quantity,
                    'cost_basis_total': total_current_cost_basis
                }

    # Convert to DataFrame for display
    final_holdings_list = [
        {'instrument': ticker, 'quantity': details['quantity'], 'cost_basis_total': details['cost_basis_total']}
        for ticker, details in current_open_positions.items()
    ]
    current_holdings_df = pd.DataFrame(final_holdings_list)
    return current_holdings_df


# --- Step 1: Connect to the Database and Load Data ---
st.header("Robinhood Portfolio Analysis (V2)")
st.markdown("---")

conn = None
try:
    conn = sqlite3.connect(DB_FILE)
    
    # Load the daily portfolio snapshots for main charts
    daily_portfolio_df = pd.read_sql_query("SELECT * FROM daily_portfolio_snapshots", conn)
    daily_portfolio_df['Date'] = pd.to_datetime(daily_portfolio_df['Date'])
    
    # Load the realized P/L summary for performance metrics
    closed_trades_df = pd.read_sql_query("SELECT * FROM closed_trades_summary", conn)

    # Load the cleaned transactions for cash flow analysis
    transactions_cleaned_df = pd.read_sql_query("SELECT * FROM transactions_cleaned", conn)
    transactions_cleaned_df['activity_date'] = pd.to_datetime(transactions_cleaned_df['activity_date'])

    st.success(f"Successfully loaded data from '{DB_FILE}'")
    
except sqlite3.Error as e:
    st.error(f"SQLite error: {e}. Please ensure '{DB_FILE}' exists.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during data loading: {e}")
    st.stop()
finally:
    if conn:
        conn.close()

# Ensure dataframes are properly prepared
daily_portfolio_df.set_index('Date', inplace=True)
daily_portfolio_df.sort_index(inplace=True)

closed_trades_df['sell_date'] = pd.to_datetime(closed_trades_df['sell_date'])

# --- Step 2: V2 Summary Metrics & Key Insights ---
st.subheader("Performance Summary (V2)")

# Calculate TWR and Drawdown from daily_portfolio_df
overall_twr = (daily_portfolio_df['cumulative_twr_factor'].iloc[-1] - 1) if not daily_portfolio_df.empty and not daily_portfolio_df['cumulative_twr_factor'].empty else 0
max_drawdown = daily_portfolio_df['drawdown'].min() if not daily_portfolio_df.empty else 0

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Overall Time-Weighted Return", f"{overall_twr * 100:.2f}%")
col2.metric("Maximum Drawdown", f"{max_drawdown * 100:.2f}%")
col3.metric("Total Realized P/L", f"${closed_trades_df['realized_profit_loss'].sum():,.2f}")

st.markdown("---")

# --- Step 3: V2 Portfolio Value Visualization ---
st.subheader("Daily Portfolio Value Over Time")
fig_portfolio_value = px.line(daily_portfolio_df, 
                               y=['Total_Portfolio_Value', 'Cash_Balance', 'Stock_Market_Value'], 
                               title='Daily Portfolio Value Components Over Time (V2)')
st.plotly_chart(fig_portfolio_value, use_container_width=True)

st.markdown("---")

# --- Step 4: V2 Drawdown Visualization ---
st.subheader("Portfolio Drawdown Over Time")
fig_drawdown = go.Figure(data=go.Scatter(
    x=daily_portfolio_df.index,
    y=daily_portfolio_df['drawdown'] * 100,
    fill='tozeroy',
    mode='lines',
    line_color='red',
    name='Drawdown'
))
fig_drawdown.update_layout(
    title='Portfolio Drawdown Over Time (V2)',
    xaxis_title='Date',
    yaxis_title='Drawdown (%)'
)
st.plotly_chart(fig_drawdown, use_container_width=True)

st.markdown("---")

# --- Step 5: V1-style Realized P/L Table & Holdings Table ---
st.subheader("Realized P/L by Instrument")
if not closed_trades_df.empty:
    realized_pl_summary = closed_trades_df.groupby('instrument')['realized_profit_loss'].sum().reset_index()
    realized_pl_summary.columns = ['Instrument', 'Total Realized P/L ($)']
    realized_pl_summary.sort_values(by='Total Realized P/L ($)', ascending=False, inplace=True)
    st.dataframe(realized_pl_summary, use_container_width=True)
else:
    st.info("No closed trades found to display realized P/L.")

st.subheader("Current Portfolio Holdings")
current_holdings_df = get_current_holdings(transactions_cleaned_df)
if not current_holdings_df.empty:
    st.dataframe(current_holdings_df.sort_values(by='quantity', ascending=False), use_container_width=True)
else:
    st.info("No current holdings found.")
