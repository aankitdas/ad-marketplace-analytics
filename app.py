"""
Ad Marketplace Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

st.set_page_config(
    page_title="Ad Marketplace Analytics",
    page_icon="📊",
    layout="wide"
)

sys.path.append('src')



# st.write("Current directory:", os.getcwd())
# st.write("Files in current directory:", os.listdir('.'))
# st.write("Data folder exists:", os.path.exists('data'))
# if os.path.exists('data'):
#     st.write("Files in data/:", os.listdir('data'))



def load_data():
    events = pd.read_csv('data/events.csv')
    daily = pd.read_csv('data/daily_metrics.csv')
    advertisers = pd.read_csv('data/advertisers.csv')
    return events, daily, advertisers

def main():
    st.title("📊 Ad Marketplace Analytics Platform")
    st.markdown("*Demonstrating marketplace simulation, causal inference, and churn prediction*")
    
    # Load data
    events, daily, advertisers = load_data()
    
    # #DEBUG
    # st.write("Events shape:", events.shape)
    # st.write("Events columns:", list(events.columns))
    # st.write("Daily shape:", daily.shape)
    # st.write("First row of daily:", daily.head(1))

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Analysis",
        ["Overview", "Experiment Analysis", "Churn Analysis", "Raw Data"]
    )
    
    if page == "Overview":
        show_overview(events, daily, advertisers)
    elif page == "Experiment Analysis":
        show_experiment(events, daily)
    elif page == "Churn Analysis":
        show_churn(advertisers, events)
    else:
        show_raw_data(events, daily, advertisers)


def show_overview(events, daily, advertisers):
    st.header("Marketplace Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Impressions", f"{len(events):,}")
    with col2:
        st.metric("Total Revenue", f"${events['price_paid'].sum():,.2f}")
    with col3:
        st.metric("Total Advertisers", len(advertisers))
    with col4:
        churn_rate = advertisers['is_churned'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    st.markdown("---")
    
    # Daily revenue chart - using Streamlit native
    st.subheader("Daily Revenue Over Time")
    st.caption("📍 Experiment Period: Days 30-60")
    st.line_chart(daily.set_index('day')['revenue'])
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Active Advertisers Over Time")
        st.line_chart(daily.set_index('day')['active_advertisers'])
    
    with col2:
        st.subheader("Advertiser Verticals")
        vertical_counts = advertisers['vertical'].value_counts()
        st.bar_chart(vertical_counts)


def show_experiment(events, daily):
    st.header("🧪 Experiment Analysis")
    st.markdown("""
    **Experiment Design:**
    - **Treatment:** Lower reserve price ($0.05 vs $0.10)
    - **Period:** Days 30-60
    - **Goal:** Measure impact on revenue and CTR
    """)
    
    # Filter experiment data
    exp_data = events[events['group'].isin(['treatment', 'control'])]
    
    # Summary metrics
    treatment = exp_data[exp_data['group'] == 'treatment']
    control = exp_data[exp_data['group'] == 'control']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Treatment Revenue", 
            f"${treatment['price_paid'].sum():,.2f}",
            f"+${treatment['price_paid'].sum() - control['price_paid'].sum():,.2f}"
        )
    with col2:
        lift = (treatment['price_paid'].sum() / control['price_paid'].sum() - 1) * 100
        st.metric("Revenue Lift", f"{lift:.1f}%")
    with col3:
        ctr_lift = (treatment['clicked'].mean() / control['clicked'].mean() - 1) * 100
        st.metric("CTR Lift", f"{ctr_lift:.1f}%")
    
    st.markdown("---")
    
    # Statistical results
    st.subheader("Statistical Analysis")
    
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(
        treatment['price_paid'], 
        control['price_paid']
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("T-Statistic", f"{t_stat:.2f}")
    with col2:
        st.metric("P-Value", f"{p_value:.4f}", delta="Significant!" if p_value < 0.05 else "Not Significant")
    
    # Revenue Distribution - using native histogram
    # Revenue Distribution - combined bar chart
    st.subheader("Revenue Distribution: Treatment vs Control")
    
    comparison_data = pd.DataFrame({
        'Total Revenue': [treatment['price_paid'].sum(), control['price_paid'].sum()],
        'Total Clicks': [treatment['clicked'].sum(), control['clicked'].sum()],
    }, index=['Treatment', 'Control'])
    st.bar_chart(comparison_data)
    
    # Daily comparison during experiment
    st.subheader("Daily Revenue by Group (During Experiment)")
    exp_daily = exp_data.groupby(['day', 'group'])['price_paid'].sum().unstack()
    st.line_chart(exp_daily)


def show_churn(advertisers, events):
    st.header("📉 Churn Analysis")
    
    # Churn stats
    churned = advertisers[advertisers['is_churned']]
    active = advertisers[~advertisers['is_churned']]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Churned", len(churned))
    with col2:
        st.metric("Still Active", len(active))
    with col3:
        st.metric("Churn Rate", f"{len(churned)/len(advertisers)*100:.1f}%")
    
    st.markdown("---")
    
    # Churn by vertical
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Rate by Vertical")
        churn_by_vertical = advertisers.groupby('vertical')['is_churned'].mean().sort_values(ascending=False) * 100
        st.bar_chart(churn_by_vertical)
    
    with col2:
        st.subheader("Churn by Budget Tier")
        advertisers_copy = advertisers.copy()
        advertisers_copy['budget_tier'] = pd.cut(
            advertisers_copy['daily_budget'], 
            bins=[0, 100, 500, 1000, float('inf')],
            labels=['$0-100', '$100-500', '$500-1000', '$1000+']
        )
        churn_by_budget = advertisers_copy.groupby('budget_tier')['is_churned'].mean() * 100
        st.bar_chart(churn_by_budget)
    
    # Churn timing
    st.subheader("When Do Advertisers Churn?")
    churn_days = churned['churn_day'].dropna()
    churn_counts = churn_days.value_counts().sort_index()
    st.bar_chart(churn_counts)
    
    # Feature importance (simplified)
    st.subheader("Churn Prediction Model")
    st.markdown("""
    **Model Performance:** AUC-ROC = 0.634
    
    **Top Predictors of Churn:**
    1. Active Days
    2. Average Price Paid
    3. Quality Score
    4. Daily Budget
    5. Signup Day
    """)


def show_raw_data(events, daily, advertisers):
    st.header("📋 Raw Data Explorer")
    
    tab1, tab2, tab3 = st.tabs(["Events", "Daily Metrics", "Advertisers"])
    
    with tab1:
        st.subheader("Event Data")
        st.write(f"Shape: {events.shape}")
        st.dataframe(events.head(1000))
    
    with tab2:
        st.subheader("Daily Metrics")
        st.write(f"Shape: {daily.shape}")
        st.dataframe(daily)
    
    with tab3:
        st.subheader("Advertiser Data")
        st.write(f"Shape: {advertisers.shape}")
        st.dataframe(advertisers)


if __name__ == '__main__':
    main()