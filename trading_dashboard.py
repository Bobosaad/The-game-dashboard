import streamlit as st
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import pytz
import plotly.graph_objects as go
import plotly.express as px
import pytz
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")

# Print current time
current_time = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.markdown(f"**Current Date and Time (UTC):** {current_time}")
st.sidebar.markdown("**User:** Bobosaad")

def simulate_trading_session(
    starting_capital: float,
    profit_target: float,
    loss_limit: float,
    daily_risk_limit: float,
    win_rate: float,
    zb_trades_per_day: int,
    gold_trades_per_day: int,
    zb_win: float,
    zb_loss: float,
    gold_win: float,
    gold_loss: float,
    zb_contracts: int,
    gold_contracts: int,
    max_trading_days: int = 90,
    max_consecutive_losses: int = 5,
    position_reduction: float = 0.5
) -> Tuple[float, List[float], int, int, int, List[bool]]:
    
    capital = starting_capital
    trading_days = 0
    daily_returns = []
    current_losing_streak = 0
    current_winning_streak = 0
    max_losing_streak = 0
    max_winning_streak = 0
    all_trades_results = []
    
    zb_position_multiplier = 1.0
    gold_position_multiplier = 1.0
    
    while trading_days < max_trading_days:
        trading_days += 1
        daily_trades_result = []
        
        total_trades = zb_trades_per_day + gold_trades_per_day
        for _ in range(total_trades):
            is_zb = len(daily_trades_result) < zb_trades_per_day
            
            if np.random.random() < win_rate:
                if is_zb:
                    trade_result = zb_win * zb_contracts * zb_position_multiplier
                else:
                    trade_result = gold_win * gold_contracts * gold_position_multiplier
                all_trades_results.append(True)
                current_winning_streak += 1
                current_losing_streak = 0
            else:
                if is_zb:
                    trade_result = zb_loss * zb_contracts * zb_position_multiplier
                else:
                    trade_result = gold_loss * gold_contracts * gold_position_multiplier
                all_trades_results.append(False)
                current_losing_streak += 1
                current_winning_streak = 0
            
            max_winning_streak = max(max_winning_streak, current_winning_streak)
            max_losing_streak = max(max_losing_streak, current_losing_streak)
            daily_trades_result.append(trade_result)
        
        daily_pnl = sum(daily_trades_result)
        if daily_pnl < -daily_risk_limit:
            daily_pnl = -daily_risk_limit
        
        capital += daily_pnl
        daily_returns.append(daily_pnl)
        
        if capital >= starting_capital + profit_target or capital <= starting_capital - loss_limit:
            break
    
    return capital, daily_returns, trading_days, max_losing_streak, max_winning_streak, all_trades_results

def run_monte_carlo(
    starting_capital: float,
    profit_target: float,
    loss_limit: float,
    daily_risk_limit: float,
    win_rate: float,
    zb_trades_per_day: int,
    gold_trades_per_day: int,
    zb_win: float,
    zb_loss: float,
    gold_win: float,
    gold_loss: float,
    zb_contracts: int,
    gold_contracts: int,
    num_simulations: int
) -> Tuple[List[float], List[float], List[int], List[int], List[int], List[bool]]:
    
    final_capitals = []
    all_daily_returns = []
    days_to_complete = []
    max_losing_streaks = []
    max_winning_streaks = []
    all_trades = []
    
    simulation_params = {
        'starting_capital': starting_capital,
        'profit_target': profit_target,
        'loss_limit': loss_limit,
        'daily_risk_limit': daily_risk_limit,
        'win_rate': win_rate,
        'zb_trades_per_day': zb_trades_per_day,
        'gold_trades_per_day': gold_trades_per_day,
        'zb_win': zb_win,
        'zb_loss': zb_loss,
        'gold_win': gold_win,
        'gold_loss': gold_loss,
        'zb_contracts': zb_contracts,
        'gold_contracts': gold_contracts
    }
    
    for _ in range(num_simulations):
        final_cap, daily_returns, days, max_lose_streak, max_win_streak, trades = simulate_trading_session(**simulation_params)
        final_capitals.append(final_cap)
        all_daily_returns.extend(daily_returns)
        days_to_complete.append(days)
        max_losing_streaks.append(max_lose_streak)
        max_winning_streaks.append(max_win_streak)
        all_trades.extend(trades)
    
    return final_capitals, all_daily_returns, days_to_complete, max_losing_streaks, max_winning_streaks, all_trades

def calculate_risk_of_ruin(final_capitals: List[float], starting_capital: float, max_loss: float) -> float:
    ruin_level = starting_capital - max_loss
    ruin_count = sum(1 for cap in final_capitals if cap <= ruin_level)
    return ruin_count / len(final_capitals)

def calculate_streak_stats(trades):
    current_streak = 0
    current_type = None
    streaks = {'winning': [], 'losing': []}
    
    for trade in trades:
        if current_type is None:
            current_type = trade
            current_streak = 1
        elif trade == current_type:
            current_streak += 1
        else:
            if current_type:
                streaks['winning'].append(current_streak)
            else:
                streaks['losing'].append(current_streak)
            current_type = trade
            current_streak = 1
    
    if current_type is not None:
        if current_type:
            streaks['winning'].append(current_streak)
        else:
            streaks['losing'].append(current_streak)
    
    return streaks

# Title and description
st.title("Trading Strategy Monte Carlo Simulation Dashboard")
st.markdown("Multi-Instrument Trading Strategy Analysis")

# Sidebar configurations
with st.sidebar:
    st.markdown("**Current Date and Time (UTC):** 2025-01-12 18:20:29")
    st.markdown("**User:** Bobosaad")
    st.markdown("---")
    
    st.header("Trading Parameters")
    
    # Monte Carlo Parameters
    with st.expander("Monte Carlo Settings", expanded=True):
        num_simulations = st.number_input("Number of Simulations", value=10000, min_value=1000, max_value=50000)
        starting_capital = st.number_input("Starting Capital ($)", value=50000, min_value=1000)
        profit_target = st.number_input("Profit Target ($)", value=3000, min_value=100)
        loss_limit = st.number_input("Loss Limit ($)", value=2000, min_value=100)
        daily_risk_limit = st.number_input("Daily Risk Limit ($)", value=500, min_value=100)
        win_rate = st.slider("Win Rate (%)", min_value=0, max_value=100, value=40) / 100

    # Contract Specifications
    with st.expander("Contract Specifications", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ZB Contract")
            zb_tick_value = st.number_input("ZB Tick Value ($)", value=31.25)
            zb_ticks_per_point = st.number_input("ZB Ticks per Point", value=32)
            zb_contracts = st.number_input("Number of ZB Contracts", value=1, min_value=1)
        
        with col2:
            st.subheader("Gold Contract")
            gold_tick_value = st.number_input("Gold Tick Value ($)", value=10.0)
            gold_ticks_per_point = st.number_input("Gold Ticks per Point", value=10)
            gold_contracts = st.number_input("Number of Gold Contracts", value=1, min_value=1)

    # Trading Parameters
    with st.expander("Trading Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            trades_per_day_zb = st.number_input("ZB Trades per Day", value=4, min_value=1)
            zb_win = st.number_input("ZB Win Amount ($)", value=183.94)
            zb_loss = st.number_input("ZB Loss Amount ($)", value=-95.53)
        
        with col2:
            trades_per_day_gold = st.number_input("Gold Trades per Day", value=4, min_value=1)
            gold_win = st.number_input("Gold Win Amount ($)", value=173.52)
            gold_loss = st.number_input("Gold Loss Amount ($)", value=-93.24)

# Trading Rules Table
st.header("Trading Rules and Risk Parameters")
trading_rules_data = {
    'Parameter': [
        'Starting Capital',
        'Daily Risk Limit',
        'Maximum Loss Limit',
        'Profit Target',
        'ZB Contracts',
        'Gold Contracts',
        'ZB Trades per Day',
        'Gold Trades per Day',
        'Win Rate',
        'ZB Win Amount',
        'ZB Loss Amount',
        'Gold Win Amount',
        'Gold Loss Amount',
        'ZB Tick Value',
        'ZB Ticks per Point',
        'Gold Tick Value',
        'Gold Ticks per Point'
    ],
    'Value': [
        f"${starting_capital:,}",
        f"${daily_risk_limit:,}",
        f"${loss_limit:,}",
        f"${profit_target:,}",
        zb_contracts,
        gold_contracts,
        trades_per_day_zb,
        trades_per_day_gold,
        f"{win_rate:.1%}",
        f"${zb_win:,.2f}",
        f"${zb_loss:,.2f}",
        f"${gold_win:,.2f}",
        f"${gold_loss:,.2f}",
        f"${zb_tick_value:,.2f}",
        zb_ticks_per_point,
        f"${gold_tick_value:,.2f}",
        gold_ticks_per_point
    ]
}

trading_rules_df = pd.DataFrame(trading_rules_data)
st.table(trading_rules_df)

# Create parameters dictionary
PARAMS = {
    'num_simulations': num_simulations,
    'starting_capital': starting_capital,
    'profit_target': profit_target,
    'loss_limit': loss_limit,
    'daily_risk_limit': daily_risk_limit,
    'win_rate': win_rate,
    'zb_trades_per_day': trades_per_day_zb,
    'gold_trades_per_day': trades_per_day_gold,
    'zb_win': zb_win,
    'zb_loss': zb_loss,
    'gold_win': gold_win,
    'gold_loss': gold_loss,
    'zb_contracts': zb_contracts,
    'gold_contracts': gold_contracts
}

# Main dashboard layout
if st.button("Run Monte Carlo Simulation"):
    with st.spinner("Running simulation..."):
        # Run simulation
        final_capitals, all_daily_returns, days_to_complete, max_losing_streaks, max_winning_streaks, all_trades = run_monte_carlo(**PARAMS)

        # Calculate statistics
        final_capitals = np.array(final_capitals)
        successes = sum(1 for cap in final_capitals if cap >= PARAMS['starting_capital'] + PARAMS['profit_target'])
        probability = successes / PARAMS['num_simulations']
        
        # Calculate risk of ruin
        risk_of_ruin = calculate_risk_of_ruin(final_capitals, PARAMS['starting_capital'], PARAMS['loss_limit'])
        
        # Calculate streak statistics
        streaks = calculate_streak_stats(all_trades)
        avg_winning_streak = np.mean(streaks['winning']) if streaks['winning'] else 0
        avg_losing_streak = np.mean(streaks['losing']) if streaks['losing'] else 0

        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Success Probability", f"{probability:.2%}")
            st.metric("Expected Value", f"${np.mean(final_capitals) - PARAMS['starting_capital']:,.2f}")
        with col2:
            st.metric("Risk of Ruin", f"{risk_of_ruin:.2%}")
            st.metric("Average Trading Days", f"{np.mean(days_to_complete):.1f}")
        with col3:
            st.metric("Average Final Capital", f"${np.mean(final_capitals):,.2f}")
            st.metric("Daily Risk Limit", f"${daily_risk_limit:,.2f}")

        # Create visualizations using plotly
        st.subheader("Distribution Analysis")

        # Final Capital Distribution
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=final_capitals,
            nbinsx=50,
            name="Final Capital Distribution",
            marker_color='blue'
        ))
        fig1.update_layout(
            title="Distribution of Final Capital",
            xaxis_title="Final Capital ($)",
            yaxis_title="Frequency",
            showlegend=True
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Daily Returns Distribution
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=all_daily_returns,
            nbinsx=50,
            name="Daily Returns Distribution",
            marker_color='green'
        ))
        fig2.update_layout(
            title="Distribution of Daily Returns",
            xaxis_title="Daily Return ($)",
            yaxis_title="Frequency",
            showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Trading Days Distribution
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=days_to_complete,
            nbinsx=50,
            name="Trading Days Distribution",
            marker_color='orange'
        ))
        fig3.update_layout(
            title="Distribution of Trading Days",
            xaxis_title="Number of Days",
            yaxis_title="Frequency",
            showlegend=True
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Streak Analysis
        st.subheader("Streak Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Winning Streak", f"{avg_winning_streak:.1f}")
        with col2:
            st.metric("Max Winning Streak", f"{max(max_winning_streaks)}")
        with col3:
            st.metric("Avg Losing Streak", f"{avg_losing_streak:.1f}")
        with col4:
            st.metric("Max Losing Streak", f"{max(max_losing_streaks)}")

        # Percentile Analysis
        st.subheader("Percentile Analysis")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(final_capitals, percentiles)
        
        percentile_df = pd.DataFrame({
            'Percentile': [f"{p}%" for p in percentiles],
            'Value': [f"${v:,.2f}" for v in percentile_values]
        })
        st.table(percentile_df)

# Footer
st.markdown("---")
st.markdown("*Trading Strategy Analysis Dashboard - Last Updated: 2025-01-12 18:20:29 UTC*")
