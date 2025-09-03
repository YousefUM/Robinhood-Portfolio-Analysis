import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from collections import deque
import numpy as np
import yfinance as yf

# --- Page Configuration (Set at the very top) ---
st.set_page_config(layout="wide", page_title="Robinhood Portfolio Analysis")

# --- Configuration ---
DB_FILE = 'robinhood_portfolio.db'

# --- Caching Functions for Live Data ---
@st.cache_data(ttl=900)  # Cache prices for 15 minutes
def get_live_prices(tickers):
    """Fetches the current market price for a list of tickers."""
    prices = {}
    if not tickers:
        return prices
    try:
        data = yf.download(tickers=tickers, period='1d', progress=False)
        if not data.empty:
            # For single ticker, yf returns a Series. For multiple, a DataFrame.
            if len(tickers) == 1:
                prices[tickers[0]] = data['Close'].iloc[-1]
            else:
                latest_prices = data['Close'].iloc[-1]
                prices = latest_prices.to_dict()
    except Exception as e:
        st.toast(f"Could not fetch some live prices: {e}", icon="⚠️")
    return prices

@st.cache_data(ttl=3600)  # Cache yields for 1 hour
def get_live_dividend_yields(tickers):
    """Fetches the current dividend yield for a list of tickers."""
    yields = {}
    if not tickers:
        return yields
    try:
        tickers_data = yf.Tickers(tickers)
        for ticker_symbol in tickers:
            info = tickers_data.tickers[ticker_symbol.upper()].info
            yield_value = info.get('dividendYield', 0)
            yields[ticker_symbol] = yield_value if yield_value is not None else 0
    except Exception as e:
        st.toast(f"Could not fetch some dividend data: {e}", icon="⚠️")
    return yields

# --- Main App ---
st.title("Robinhood Portfolio Analysis 📈")
st.markdown("---")

# --- Data Loading ---
try:
    conn = sqlite3.connect(DB_FILE)
    # Load all necessary tables from the database
    daily_portfolio_df = pd.read_sql_query("SELECT * FROM daily_portfolio_snapshots", conn)
    closed_trades_df = pd.read_sql_query("SELECT * FROM closed_trades_summary", conn)
    transactions_cleaned_df = pd.read_sql_query("SELECT * FROM transactions_cleaned", conn)
    # NEW: Load from the simplified open positions table
    open_positions_df = pd.read_sql_query("SELECT * FROM current_holdings_summary", conn)
    conn.close()

    # Standard date conversions
    daily_portfolio_df['Date'] = pd.to_datetime(daily_portfolio_df['Date'])
    daily_portfolio_df.set_index('Date', inplace=True)
    daily_portfolio_df.sort_index(inplace=True)
    closed_trades_df['sell_date'] = pd.to_datetime(closed_trades_df['sell_date'])
    transactions_cleaned_df['activity_date'] = pd.to_datetime(transactions_cleaned_df['activity_date'])

except Exception as e:
    st.error(f"An error occurred loading the database: {e}. Please ensure '{DB_FILE}' exists and the data pipeline has been run successfully.")
    st.stop()

# --- Key Metrics Display ---
st.subheader("Performance Summary")
# ... (This section remains unchanged) ...
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
tab_main, tab_analysis, tab_cash_flow = st.tabs(["Main Dashboard", "Detailed Analysis", "Cash Flow"])

with tab_main:
    st.subheader("Current Portfolio Holdings (Live)")

    if not open_positions_df.empty:
        # Create a copy to work with
        holdings_df = open_positions_df.copy()
        
        # --- LOGIC MOVED FROM NOTEBOOK TO APP ---
        # 1. Fetch live prices and dividend yields
        all_tickers = holdings_df['instrument'].tolist()
        live_prices = get_live_prices(all_tickers)
        dividend_yields = get_live_dividend_yields(all_tickers)

        # 2. Map live data to the DataFrame
        holdings_df['current_price'] = holdings_df['instrument'].map(live_prices).fillna(0)
        holdings_df['dividend_yield'] = holdings_df['instrument'].map(dividend_yields).fillna(0)

        # 3. Perform calculations using live data
        holdings_df['market_value'] = holdings_df['quantity'] * holdings_df['current_price']
        holdings_df['unrealized_pl'] = holdings_df['market_value'] - holdings_df['cost_basis_total']
        holdings_df['avg_cost_price'] = (holdings_df['cost_basis_total'] / holdings_df['quantity']).replace([np.inf, -np.inf], 0)
        holdings_df['unrealized_pl_pct'] = (holdings_df['unrealized_pl'] / holdings_df['cost_basis_total'].replace(0, np.nan)).fillna(0) * 100
        
        total_market_value = holdings_df['market_value'].sum()
        if total_market_value > 0:
            holdings_df['portfolio_allocation_pct'] = (holdings_df['market_value'] / total_market_value) * 100
        else:
            holdings_df['portfolio_allocation_pct'] = 0
        # --- END OF MOVED LOGIC ---

        # Display the DataFrame with live data
        st.dataframe(
            holdings_df[['instrument', 'quantity', 'avg_cost_price', 'current_price', 'market_value', 'dividend_yield', 'unrealized_pl', 'unrealized_pl_pct', 'portfolio_allocation_pct']].sort_values(by='market_value', ascending=False),
            column_config={
                "instrument": st.column_config.TextColumn("Instrument"),
                "quantity": st.column_config.NumberColumn(format="%.4f"),
                "avg_cost_price": st.column_config.NumberColumn("Avg Cost", format="$%.2f"),
                "current_price": st.column_config.NumberColumn("Price (Live)", format="$%.2f"),
                "market_value": st.column_config.NumberColumn("Market Value (Live)", format="$%,.2f"),
                "dividend_yield": st.column_config.NumberColumn("Yield (Live)", format="%.2f%%"),
                "unrealized_pl": st.column_config.NumberColumn("Unrealized P/L (Live)", format="$%,.2f"),
                "unrealized_pl_pct": st.column_config.NumberColumn("Unrealized P/L % (Live)", format="%.2f%%"),
                "portfolio_allocation_pct": st.column_config.NumberColumn("Allocation %", format="%.2f%%"),
            },
            use_container_width=True,
            height=35 * (len(holdings_df) + 1)
        )
    else:
        st.info("No current holdings found in the database.")

    # ... (The rest of the app, including charts and other tabs, remains the same) ...
    st.subheader("Portfolio Charts")
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Performance vs. Benchmark", "Drawdown", "Portfolio Value"])
    with chart_tab1:
        if 'benchmark_cumulative_return' in daily_portfolio_df.columns:
            fig_benchmark = go.Figure()
            fig_benchmark.add_trace(go.Scatter(x=daily_portfolio_df.index, y=daily_portfolio_df['cumulative_twr_factor'], mode='lines', name='My Portfolio', line=dict(color='royalblue', width=2)))
            fig_benchmark.add_trace(go.Scatter(x=daily_portfolio_df.index, y=daily_portfolio_df['benchmark_cumulative_return'], mode='lines', name='S&P 500 (^GSPC)', line=dict(color='grey', width=2, dash='dash')))
            fig_benchmark.update_layout(title='Cumulative Growth: Portfolio vs. Benchmark', xaxis_title='Date', yaxis_title='Cumulative Return Factor (Growth of $1)', legend_title='Legend')
            st.plotly_chart(fig_benchmark, use_container_width=True)
        else:
            st.info("Benchmark comparison data not found.")

    with chart_tab2:
        fig_drawdown = go.Figure(data=go.Scatter(x=daily_portfolio_df.index, y=daily_portfolio_df['drawdown'] * 100, fill='tozeroy', mode='lines', line_color='red', name='Drawdown'))
        fig_drawdown.update_layout(title='Portfolio Drawdown Over Time', xaxis_title='Date', yaxis_title='Drawdown (%)')
        st.plotly_chart(fig_drawdown, use_container_width=True)

    with chart_tab3:
        fig_portfolio_value = px.line(daily_portfolio_df, y=['Total_Portfolio_Value', 'Cash_Balance', 'Stock_Market_Value'], title='Daily Portfolio Value Components Over Time')
        st.plotly_chart(fig_portfolio_value, use_container_width=True)

with tab_analysis:
    st.subheader("Trading Performance Insights")
    if not closed_trades_df.empty:
        winning_trades = closed_trades_df[closed_trades_df['realized_profit_loss'] > 0]
        losing_trades = closed_trades_df[closed_trades_df['realized_profit_loss'] < 0]
        total_trades = len(closed_trades_df)
        win_count = len(winning_trades)
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
        avg_win = winning_trades['realized_profit_loss'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['realized_profit_loss'].mean() if not losing_trades.empty else 0
        total_gains = winning_trades['realized_profit_loss'].sum()
        total_losses = abs(losing_trades['realized_profit_loss'].sum())
        
        if total_losses > 0:
            profit_factor = total_gains / total_losses
        elif total_gains > 0:
            profit_factor = float('inf')
        else:
            profit_factor = 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Win Rate", f"{win_rate:.2f}%", help="The percentage of closed trades that were profitable.")
        col2.metric("Avg. Gain / Loss", f"${avg_win:,.2f} / ${abs(avg_loss):,.2f}", help="The average dollar amount for winning and losing trades.")
        col3.metric("Profit Factor", f"{profit_factor:.2f}", help="Total gains divided by total losses. A value > 1 indicates profitability.")
    else:
        st.info("No closed trades to analyze.")
    
    st.markdown("---")
    st.subheader("Portfolio Sector Allocation")
    try:
        conn = sqlite3.connect(DB_FILE)
        sector_df = pd.read_sql_query("SELECT * FROM instrument_sectors", conn)
        conn.close()
        
        # Use open_positions_df for sector analysis now
        if not open_positions_df.empty and not sector_df.empty:
            holdings_with_sector = pd.merge(open_positions_df, sector_df, on='instrument', how='left')
            holdings_with_sector['sector'].fillna('N/A', inplace=True)
            sector_allocation = holdings_with_sector.groupby('sector')['cost_basis_total'].sum().reset_index()
            
            fig_pie = px.pie(sector_allocation, names='sector', values='cost_basis_total', title='Sector Allocation by Cost Basis', hole=0.3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Not enough data to generate sector allocation chart.")
    except Exception as e:
        st.warning(f"Could not generate sector analysis: {e}")

    st.markdown("---")
    st.subheader("Instrument-Specific Analysis")
    all_instruments = sorted(transactions_cleaned_df['instrument'].dropna().unique().tolist())
    selected_instrument = st.selectbox("Select an Instrument to Analyze", all_instruments)

    if selected_instrument:
        instrument_transactions = transactions_cleaned_df[transactions_cleaned_df['instrument'] == selected_instrument]
        instrument_closed_trades = closed_trades_df[closed_trades_df['instrument'] == selected_instrument]
        
        st.markdown(f"#### Performance Metrics for **{selected_instrument}**")
        total_realized_pl = instrument_closed_trades['realized_profit_loss'].sum()
        win_count = instrument_closed_trades[instrument_closed_trades['realized_profit_loss'] > 0].shape[0]
        loss_count = instrument_closed_trades[instrument_closed_trades['realized_profit_loss'] < 0].shape[0]
        total_closed_trades = win_count + loss_count
        win_rate_instrument = (win_count / total_closed_trades) * 100 if total_closed_trades > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Realized P/L", f"${total_realized_pl:,.2f}")
        col2.metric("Win Rate", f"{win_rate_instrument:.2f}%")
        col3.metric("Closed Trades", f"{total_closed_trades}")
        
        st.markdown("##### Transaction History")
        st.dataframe(instrument_transactions[['activity_date', 'trans_code', 'quantity', 'price', 'amount']].sort_values(by='activity_date', ascending=False), use_container_width=True)
        
        if not instrument_closed_trades.empty:
            st.markdown("##### Closed Trades Summary")
            st.dataframe(instrument_closed_trades[['sell_date', 'sold_quantity_transaction', 'sell_price', 'realized_profit_loss', 'holding_period_days']].sort_values(by='sell_date', ascending=False), use_container_width=True)

with tab_cash_flow:
    st.subheader("Cash Flow Analysis 💰")
    # ... (This tab remains unchanged) ...
    st.markdown("This section analyzes all non-trade financial activities, such as dividends, fees, and deposits/withdrawals.")
    cash_flow_categories = ['Income', 'Expense', 'Cash_Movement', 'Corporate_Action', 'Cash_Adjustment']
    non_trade_cash_flows_df = transactions_cleaned_df[transactions_cleaned_df['transaction_category'].isin(cash_flow_categories)].copy()
    if not non_trade_cash_flows_df.empty:
        total_income = non_trade_cash_flows_df[non_trade_cash_flows_df['transaction_category'] == 'Income']['amount'].sum()
        total_expense = non_trade_cash_flows_df[non_trade_cash_flows_df['transaction_category'] == 'Expense']['amount'].sum()
        net_cash_movement = non_trade_cash_flows_df[non_trade_cash_flows_df['transaction_category'] == 'Cash_Movement']['amount'].sum()
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total Income", f"${total_income:,.2f}", help="All income from dividends (CDIV, MDIV), interest (INT), and other sources (REC).")
        metric_col2.metric("Total Expenses", f"${total_expense:,.2f}", help="All expenses from fees (DFEE, AFEE), taxes (DTAX), and subscriptions (GOLD).")
        metric_col3.metric("Net Deposits", f"${net_cash_movement:,.2f}", help="Total cash deposited into the account minus total cash withdrawn.")
        st.markdown("---")
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.markdown("##### Income Sources")
            income_df = non_trade_cash_flows_df[non_trade_cash_flows_df['transaction_category'] == 'Income']
            if not income_df.empty:
                income_by_type = income_df.groupby('trans_code')['amount'].sum().reset_index()
                fig_income = px.pie(income_by_type, names='trans_code', values='amount', title='Income Breakdown by Type', hole=0.4)
                fig_income.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_income, use_container_width=True)
            else:
                st.info("No income data to display.")
        with viz_col2:
            st.markdown("##### Expense Sources")
            expense_df = non_trade_cash_flows_df[non_trade_cash_flows_df['transaction_category'] == 'Expense']
            if not expense_df.empty:
                expense_by_type = expense_df.groupby('trans_code')['amount'].sum().abs().reset_index()
                fig_expense = px.pie(expense_by_type, names='trans_code', values='amount', title='Expense Breakdown by Type', hole=0.4)
                fig_expense.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_expense, use_container_width=True)
            else:
                st.info("No expense data to display.")
        st.markdown("---")
        st.subheader("Cumulative Net Cash Flow Over Time")
        daily_cash_flow = non_trade_cash_flows_df.groupby(non_trade_cash_flows_df['activity_date'].dt.date)['amount'].sum().reset_index()
        daily_cash_flow.columns = ['Date', 'Daily_Net_Cash_Flow']
        daily_cash_flow.sort_values('Date', inplace=True)
        daily_cash_flow['Cumulative_Cash_Flow'] = daily_cash_flow['Daily_Net_Cash_Flow'].cumsum()
        fig_cumulative_cash = px.area(daily_cash_flow, x='Date', y='Cumulative_Cash_Flow', title='Cumulative Net Cash Flow (Non-Trade Activities)')
        fig_cumulative_cash.update_layout(xaxis_title='Date', yaxis_title='Cumulative Cash Flow ($)')
        st.plotly_chart(fig_cumulative_cash, use_container_width=True)
    else:
        st.warning("No cash flow transaction data found to generate the analysis.")
