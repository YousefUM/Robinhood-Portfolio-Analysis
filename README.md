# Robinhood-Portfolio-Analysis

## Project Overview
This project is a comprehensive data analysis tool designed to ingest, clean, and analyze a user's Robinhood transaction history. The goal is to provide accurate financial metrics and insightful visualizations that go beyond the basic reports offered by Robinhood.

This repository contains the complete codebase for **Version 2**, featuring a robust data pipeline and a powerful Streamlit web application.

🔗 **Try the live app here:** [Robinhood Portfolio Analysis](https://robinhood-portfolio-analysis.streamlit.app/)

---

## Key Features

### 📊 Data Processing & Accuracy
- **Robust Data Cleaning**: Ingests raw CSV data and performs rigorous cleaning, type conversions, and error handling.  
- **Database Integration**: Uses an SQLite database (`robinhood_portfolio.db`) to store raw, cleaned, and processed data.  
- **Corporate Action Handling**: Accurately processes complex corporate actions, including:  
  - Splits & Reverse Splits (`SPL`, `SPR`)  
  - Mergers & Conversions (`MRGS`, `CONV`, `SXCH`)  
  - Spin-offs (`SOFF`)  
  - Stock Gifts (`REC`)  
- **Live Market Data**: Fetches historical daily closing prices from Yahoo Finance (`yfinance`) to enable accurate portfolio valuation.  

### 💰 Financial Analysis & Metrics
- **Realized Profit/Loss (P&L)**: Calculates P&L for all closed trades using the First-In, First-Out (FIFO) method.  
- **Current Holdings**: Provides a list of all open positions with cost basis (FIFO).  
- **Daily Portfolio Snapshots**: Reconstructs daily portfolio history (cash balance, stock market value, and total portfolio value).  
- **Time-Weighted Return (TWR)**: Computes the true rate of return, adjusted for deposits, withdrawals, and fees.  
- **Maximum Drawdown**: Measures the largest peak-to-trough decline, a critical risk metric.  

### 📈 Interactive Visualizations
The Streamlit app turns raw data into interactive visual insights:  
- **Portfolio Value Over Time**: Dynamic chart showing portfolio value, cash balance, and stock market value.  
- **Portfolio Drawdown**: Visual representation of portfolio risk and historical declines.  

---

## Tech Stack
- **Python** (pandas, numpy, yfinance, sqlite3)  
- **Streamlit** (for interactive dashboards)  
- **SQLite** (for structured data storage)  

---

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/YousefUM/Robinhood-Portfolio-Analysis.git
   cd Robinhood-Portfolio-Analysis
