"""
Ad Marketplace Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append('src')

st.set_page_config(
    page_title="Ad Marketplace Analytics",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    # Check if data exists
    if not os.path.exists('data/events.parquet'):
        st.warning("Generating data... this takes ~2 minutes on first run")
        generate_data()
    
    events = pd.read_parquet('data/events.parquet')
    daily = pd.read_parquet('data/daily_metrics.parquet')
    advertisers = pd.read_parquet('data/advertisers.parquet')
    return events, daily, advertisers

def generate_data():
    """Generate simulation data if it doesn't exist."""
    import sys
    sys.path.append('src/simulation')
    from full_simulation import FullMarketplaceSimulator
    
    os.makedirs('data', exist_ok=True)
    
    sim = FullMarketplaceSimulator(seed=42)
    results = sim.run(n_advertisers=200, n_days=90, impressions_per_day=5000)
    
    results['events'].to_parquet('data/events.parquet', index=False)
    results['daily_metrics'].to_parquet('data/daily_metrics.parquet', index=False)
    results['advertisers'].to_parquet('data/advertisers.parquet', index=False)

def main():
    st.title("📊 Ad Marketplace Analytics Platform")
    st.markdown("*Demonstrating marketplace simulation, causal inference, and churn prediction*")
    
    # Load data
    events, daily, advertisers = load_data()
    
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
    
    # Daily revenue chart
    st.subheader("Daily Revenue Over Time")
    fig = px.line(daily, x='day', y='revenue', title='Daily Revenue')
    fig.add_vrect(x0=30, x1=60, fillcolor="green", opacity=0.1, 
                  annotation_text="Experiment Period", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Active Advertisers Over Time")
        fig = px.line(daily, x='day', y='active_advertisers')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Advertiser Verticals")
        vertical_counts = advertisers['vertical'].value_counts()
        fig = px.pie(values=vertical_counts.values, names=vertical_counts.index)
        st.plotly_chart(fig, use_container_width=True)


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
    
    # Comparison chart
    st.subheader("Revenue Distribution: Treatment vs Control")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=treatment['price_paid'], name='Treatment', opacity=0.7))
    fig.add_trace(go.Histogram(x=control['price_paid'], name='Control', opacity=0.7))
    fig.update_layout(barmode='overlay', xaxis_title='Price Paid', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)
    
    # Daily comparison during experiment
    st.subheader("Daily Revenue by Group (During Experiment)")
    exp_daily = exp_data.groupby(['day', 'group'])['price_paid'].sum().reset_index()
    fig = px.line(exp_daily, x='day', y='price_paid', color='group', 
                  title='Daily Revenue: Treatment vs Control')
    st.plotly_chart(fig, use_container_width=True)


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
        churn_by_vertical = advertisers.groupby('vertical')['is_churned'].mean().sort_values(ascending=False)
        fig = px.bar(x=churn_by_vertical.index, y=churn_by_vertical.values * 100,
                     labels={'x': 'Vertical', 'y': 'Churn Rate (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Churn by Budget Tier")
        advertisers['budget_tier'] = pd.cut(
            advertisers['daily_budget'], 
            bins=[0, 100, 500, 1000, float('inf')],
            labels=['$0-100', '$100-500', '$500-1000', '$1000+']
        )
        churn_by_budget = advertisers.groupby('budget_tier')['is_churned'].mean()
        fig = px.bar(x=churn_by_budget.index.astype(str), y=churn_by_budget.values * 100,
                     labels={'x': 'Budget Tier', 'y': 'Churn Rate (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Churn timing
    st.subheader("When Do Advertisers Churn?")
    churn_days = churned['churn_day'].dropna()
    fig = px.histogram(churn_days, nbins=30, title='Distribution of Churn Day')
    fig.update_layout(xaxis_title='Day', yaxis_title='Number of Churns')
    st.plotly_chart(fig, use_container_width=True)
    
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