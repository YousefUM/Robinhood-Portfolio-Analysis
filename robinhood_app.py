import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
DB_FILE = 'robinhood_portfolio.db'

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

# --- Placeholder for next steps ---
# You can add code here to check the loaded dataframes
# st.write(daily_portfolio_df.head())
# st.write(closed_trades_df.head())
# st.write(transactions_cleaned_df.head())

st.subheader("V2 Analysis & Visualizations")

# (All subsequent Streamlit code for charts and metrics will go here)
# --- Step 2: Calculate and Display Key Metrics ---
if not daily_portfolio_df.empty:
    # Calculate daily returns (simplified TWR method)
    # Get previous day's total portfolio value
    daily_portfolio_df['prev_day_value'] = daily_portfolio_df['Total_Portfolio_Value'].shift(1)
    
    # Calculate daily external cash flows from transactions_cleaned_df
    external_cash_flow_categories = ['Cash_Movement', 'Income', 'Expense', 'Cash_Adjustment', 'Uncategorized'] 
    daily_external_cash_flow = transactions_cleaned_df[
        transactions_cleaned_df['transaction_category'].isin(external_cash_flow_categories)
    ].groupby('activity_date')['amount'].sum()
    
    # Align cash flows with daily portfolio dataframe's index, fill missing days with 0
    daily_cash_flow_aligned = daily_external_cash_flow.reindex(daily_portfolio_df['Date'], fill_value=0).fillna(0)
    
    # Calculate daily returns, adjusting for cash flows
    daily_portfolio_df['return_base'] = daily_portfolio_df['prev_day_value'] + daily_cash_flow_aligned
    daily_portfolio_df['daily_return_adjusted'] = (daily_portfolio_df['Total_Portfolio_Value'] - daily_portfolio_df['return_base']) / daily_portfolio_df['return_base']
    daily_portfolio_df['daily_return_adjusted'].fillna(0, inplace=True)
    daily_portfolio_df.loc[daily_portfolio_df['return_base'] <= 0, 'daily_return_adjusted'] = 0
    
    # Calculate cumulative TWR factor
    daily_portfolio_df['cumulative_twr_factor'] = (1 + daily_portfolio_df['daily_return_adjusted']).cumprod()
    overall_twr = (daily_portfolio_df['cumulative_twr_factor'].iloc[-1] - 1) * 100
    
    # Calculate Maximum Drawdown
    daily_portfolio_df['peak_value'] = daily_portfolio_df['Total_Portfolio_Value'].expanding(min_periods=1).max()
    daily_portfolio_df['drawdown'] = (daily_portfolio_df['Total_Portfolio_Value'] - daily_portfolio_df['peak_value']) / daily_portfolio_df['peak_value']
    max_drawdown = daily_portfolio_df['drawdown'].min() * 100
    
    st.subheader("Portfolio Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Overall Time-Weighted Return (TWR)", value=f"{overall_twr:.2f}%")
    
    with col2:
        st.metric(label="Maximum Drawdown", value=f"{max_drawdown:.2f}%")
    
    with col3:
        latest_value = daily_portfolio_df['Total_Portfolio_Value'].iloc[-1]
        st.metric(label="Current Portfolio Value", value=f"${latest_value:,.2f}")
