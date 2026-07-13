"""
Sales Forecasting & Demand Intelligence Dashboard
Internship Final Project - Task 7 (Streamlit Deployment)

Author: Vihaan Singh

Run locally with:   streamlit run app.py
Deploy on Streamlit Community Cloud by pointing it at this repo (train.csv must sit
alongside app.py in the same folder).
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide", page_icon="📊")

# ============================================================
# DATA LOADING (cached)
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
    df['Postal Code'] = df.groupby('State')['Postal Code'].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 0))
    df['Postal Code'] = df['Postal Code'].fillna(0)
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Quarter'] = df['Order Date'].dt.quarter
    return df

df = load_data()

# Pre-defined SARIMA orders found during offline model selection (Task 3 / Task 4).
# Re-using these here (rather than re-running a full grid search inside the live app)
# keeps the dashboard fast to load; see analysis.ipynb for how each order was chosen.
SARIMA_ORDERS = {
    'Overall':          {'order': (2, 1, 2), 'seasonal_order': (1, 1, 1, 12)},
    'Furniture':        {'order': (1, 0, 2), 'seasonal_order': (1, 1, 1, 12)},
    'Technology':       {'order': (1, 0, 2), 'seasonal_order': (0, 1, 1, 12)},
    'Office Supplies':  {'order': (1, 0, 2), 'seasonal_order': (0, 1, 1, 12)},
    'West':             {'order': (1, 1, 2), 'seasonal_order': (0, 1, 1, 12)},
    'East':             {'order': (0, 0, 2), 'seasonal_order': (0, 1, 1, 12)},
    'Central':          {'order': (1, 1, 1), 'seasonal_order': (0, 1, 1, 12)},
    'South':            {'order': (1, 1, 1), 'seasonal_order': (0, 1, 1, 12)},
}

@st.cache_data
def get_monthly_series(_df, filter_type, filter_value):
    if filter_type == 'Overall':
        sub = _df
    elif filter_type == 'Category':
        sub = _df[_df['Category'] == filter_value]
    else:
        sub = _df[_df['Region'] == filter_value]
    monthly = sub.set_index('Order Date').resample('MS')['Sales'].sum().asfreq('MS').fillna(0)
    return monthly

@st.cache_resource
def fit_sarima(monthly_key, _monthly_series, order, seasonal_order):
    model = SARIMAX(_monthly_series, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)

def backtest_sarima(monthly_series, order, seasonal_order, n_test=3):
    if len(monthly_series) <= n_test + 6:
        return None, None, None
    train, test = monthly_series.iloc[:-n_test], monthly_series.iloc[-n_test:]
    fit = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred = fit.get_forecast(steps=n_test).predicted_mean
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))
    mape = np.mean(np.abs((test.values - pred.values) / test.values)) * 100
    return mae, rmse, mape

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", [
    "1. Sales Overview",
    "2. Forecast Explorer",
    "3. Anomaly Report",
    "4. Product Demand Segments",
])
st.sidebar.markdown("---")
st.sidebar.caption("End-to-End Sales Forecasting & Demand Intelligence System · "
                    "Superstore Sales dataset (2015-2018) · Model: SARIMA (production pick)")

# ============================================================
# PAGE 1: SALES OVERVIEW
# ============================================================
if page == "1. Sales Overview":
    st.title("Sales Overview Dashboard")
    st.caption("High-level view of historical performance across years, regions, and categories.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    col2.metric("Total Orders", f"{df['Order ID'].nunique():,}")
    col3.metric("Avg Order Value", f"${df.groupby('Order ID')['Sales'].sum().mean():,.2f}")
    col4.metric("Years Covered", f"{df['Order Year'].min()}-{df['Order Year'].max()}")

    st.markdown("### Total Sales by Year")
    yearly = df.groupby('Order Year')['Sales'].sum().reset_index()
    fig_year = px.bar(yearly, x='Order Year', y='Sales', text_auto='.2s',
                       color='Sales', color_continuous_scale='Blues')
    fig_year.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_year, use_container_width=True)

    st.markdown("### Monthly Sales Trend")
    monthly = df.set_index('Order Date').resample('MS')['Sales'].sum().reset_index()
    fig_month = px.line(monthly, x='Order Date', y='Sales', markers=True)
    fig_month.update_traces(line_color='#2563eb')
    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("### Sales by Region & Category (interactive filters)")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        regions = st.multiselect("Filter by Region", options=sorted(df['Region'].unique()),
                                  default=sorted(df['Region'].unique()))
    with fcol2:
        categories = st.multiselect("Filter by Category", options=sorted(df['Category'].unique()),
                                     default=sorted(df['Category'].unique()))

    filtered = df[df['Region'].isin(regions) & df['Category'].isin(categories)]
    ccol1, ccol2 = st.columns(2)
    with ccol1:
        region_sales = filtered.groupby('Region')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
        fig_region = px.pie(region_sales, names='Region', values='Sales', hole=0.4,
                             title='Sales Share by Region')
        st.plotly_chart(fig_region, use_container_width=True)
    with ccol2:
        cat_sales = filtered.groupby('Category')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
        fig_cat = px.bar(cat_sales, x='Category', y='Sales', color='Category',
                          title='Sales by Category')
        st.plotly_chart(fig_cat, use_container_width=True)

# ============================================================
# PAGE 2: FORECAST EXPLORER
# ============================================================
elif page == "2. Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("SARIMA was selected as the production model in Task 3 (lowest RMSE on a "
               "3-month holdout test versus Prophet and XGBoost). Pick a slice below to see its forecast.")

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        dim = st.selectbox("Forecast by", ["Overall", "Category", "Region"])
    with fcol2:
        if dim == "Category":
            value = st.selectbox("Select Category", sorted(df['Category'].unique()))
        elif dim == "Region":
            value = st.selectbox("Select Region", sorted(df['Region'].unique()))
        else:
            value = None

    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    key_name = value if value else 'Overall'
    monthly_series = get_monthly_series(df, dim, value)

    params = SARIMA_ORDERS.get(key_name, {'order': (1, 1, 1), 'seasonal_order': (0, 1, 1, 12)})
    with st.spinner(f"Fitting SARIMA{params['order']}x{params['seasonal_order']} on {key_name} sales..."):
        fit = fit_sarima(key_name, monthly_series, params['order'], params['seasonal_order'])
        forecast_obj = fit.get_forecast(steps=horizon)
        pred = forecast_obj.predicted_mean.clip(lower=0)
        ci = forecast_obj.conf_int(alpha=0.05).clip(lower=0)
        mae, rmse, mape = backtest_sarima(monthly_series, params['order'], params['seasonal_order'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_series.index, y=monthly_series.values,
                              name='Actual', mode='lines+markers', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=pred.index, y=pred.values,
                              name='Forecast', mode='lines+markers', line=dict(color='#2563eb', dash='dash')))
    fig.add_trace(go.Scatter(x=list(ci.index) + list(ci.index[::-1]),
                              y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
                              fill='toself', fillcolor='rgba(37,99,235,0.15)',
                              line=dict(color='rgba(255,255,255,0)'), name='95% CI'))
    fig.update_layout(title=f"SARIMA Forecast - {key_name}", xaxis_title="Month", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Forecast values")
    forecast_table = pd.DataFrame({
        'Month': pred.index.strftime('%B %Y'),
        'Forecasted Sales': pred.values.round(2),
        'Lower 95% CI': ci.iloc[:, 0].values.round(2),
        'Upper 95% CI': ci.iloc[:, 1].values.round(2),
    })
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

    st.markdown("#### Model performance (3-month holdout backtest)")
    if mae is not None:
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("MAE", f"${mae:,.0f}")
        mcol2.metric("RMSE", f"${rmse:,.0f}")
        mcol3.metric("MAPE", f"{mape:.1f}%")
    else:
        st.info("Not enough historical data in this slice to run a holdout backtest.")

# ============================================================
# PAGE 3: ANOMALY REPORT
# ============================================================
elif page == "3. Anomaly Report":
    st.title("Anomaly Report")
    st.caption("Two independent methods flag unusual sales weeks: Isolation Forest (ML-based) "
               "and a rolling Z-score (statistical). Weeks flagged by both are the highest-confidence anomalies.")

    weekly = df.set_index('Order Date').resample('W')['Sales'].sum().asfreq('W').fillna(0)
    feat = pd.DataFrame({'Sales': weekly})
    feat['rolling_mean_4'] = feat['Sales'].rolling(4, center=True, min_periods=1).mean()
    feat['rolling_std_4'] = feat['Sales'].rolling(4, center=True, min_periods=1).std().fillna(0)

    iso = IsolationForest(n_estimators=200, contamination=0.08, random_state=42)
    feat['iso_anomaly'] = iso.fit_predict(feat[['Sales', 'rolling_mean_4', 'rolling_std_4']].fillna(0)) == -1

    roll_window = 8
    feat['roll_mean_z'] = feat['Sales'].rolling(roll_window, min_periods=4).mean()
    feat['roll_std_z'] = feat['Sales'].rolling(roll_window, min_periods=4).std()
    feat['z_score'] = (feat['Sales'] - feat['roll_mean_z']) / feat['roll_std_z']
    feat['zscore_anomaly'] = feat['z_score'].abs() > 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feat.index, y=feat['Sales'], mode='lines', name='Weekly Sales',
                              line=dict(color='#94a3b8')))
    iso_pts = feat[feat['iso_anomaly']]
    fig.add_trace(go.Scatter(x=iso_pts.index, y=iso_pts['Sales'], mode='markers', name='Isolation Forest anomaly',
                              marker=dict(color='#dc2626', size=10)))
    z_pts = feat[feat['zscore_anomaly']]
    fig.add_trace(go.Scatter(x=z_pts.index, y=z_pts['Sales'], mode='markers', name='Z-score anomaly',
                              marker=dict(color='#2563eb', size=12, symbol='x')))
    fig.update_layout(title="Weekly Sales with Detected Anomalies", xaxis_title="Week", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Isolation Forest flags", int(feat['iso_anomaly'].sum()))
    m2.metric("Z-score flags", int(feat['zscore_anomaly'].sum()))
    m3.metric("Flagged by both", int((feat['iso_anomaly'] & feat['zscore_anomaly']).sum()))

    st.markdown("#### Detected anomaly weeks")
    anomalies = feat[feat['iso_anomaly'] | feat['zscore_anomaly']].copy()
    anomalies['Flagged By'] = anomalies.apply(
        lambda r: ' + '.join([n for n, f in [('Isolation Forest', r['iso_anomaly']), ('Z-score', r['zscore_anomaly'])] if f]),
        axis=1)
    anomalies_table = anomalies[['Sales', 'z_score', 'Flagged By']].reset_index().rename(columns={'Order Date': 'Week'})
    anomalies_table['Week'] = anomalies_table['Week'].dt.strftime('%Y-%m-%d')
    anomalies_table['Sales'] = anomalies_table['Sales'].round(0)
    anomalies_table['z_score'] = anomalies_table['z_score'].round(2)
    st.dataframe(anomalies_table.sort_values('z_score', key=abs, ascending=False),
                 use_container_width=True, hide_index=True)

# ============================================================
# PAGE 4: PRODUCT DEMAND SEGMENTS
# ============================================================
elif page == "4. Product Demand Segments":
    st.title("Product Demand Segments")
    st.caption("Sub-categories grouped by demand behaviour using K-Means clustering "
               "(total volume, YoY growth rate, volatility, average order value).")

    subcat_monthly = df.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='MS')])['Sales'].sum().reset_index()
    features = []
    for subcat, g in subcat_monthly.groupby('Sub-Category'):
        g = g.set_index('Order Date').asfreq('MS').fillna(0)['Sales']
        total_volume = g.sum()
        avg_order_value = df[df['Sub-Category'] == subcat]['Sales'].mean()
        volatility = g.std()
        yearly = g.groupby(g.index.year).sum()
        yoy_growth = (yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] / (len(yearly) - 1) * 100 if len(yearly) >= 2 and yearly.iloc[0] > 0 else 0
        features.append({'Sub-Category': subcat, 'Total_Sales_Volume': total_volume,
                          'YoY_Growth_Rate_pct': yoy_growth, 'Sales_Volatility': volatility,
                          'Avg_Order_Value': avg_order_value})
    feat_df = pd.DataFrame(features).set_index('Sub-Category')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    feat_df['Cluster'] = kmeans.fit_predict(X_scaled)

    centroid_stats = feat_df.groupby('Cluster')[['Total_Sales_Volume', 'YoY_Growth_Rate_pct', 'Sales_Volatility']].mean()
    growth_rank = centroid_stats['YoY_Growth_Rate_pct'].rank(ascending=False)
    volume_rank = centroid_stats['Total_Sales_Volume'].rank(ascending=False)
    volat_rank = centroid_stats['Sales_Volatility'].rank(ascending=False)
    n_clusters_actual = len(centroid_stats)
    growth_median = feat_df['YoY_Growth_Rate_pct'].median()

    def label_cluster(c):
        growth = centroid_stats.loc[c, 'YoY_Growth_Rate_pct']
        if growth < 0:
            return 'Declining Demand'
        if growth_rank[c] == 1 and volat_rank[c] == 1:
            return 'High-Growth, Volatile Niche Demand'
        if volume_rank[c] == 1 and volat_rank[c] == n_clusters_actual:
            return 'High Volume, Stable Demand'
        if volume_rank[c] == 1:
            return 'High Volume, Mature Demand (Core Revenue Driver)'
        if volume_rank[c] == n_clusters_actual and volat_rank[c] == 1:
            return 'Low Volume, High Volatility'
        if growth_rank[c] == 1 or growth >= growth_median:
            return 'Growing Demand'
        return 'Moderate, Balanced Demand'

    cluster_labels = {c: label_cluster(c) for c in centroid_stats.index}
    feat_df['Cluster_Label'] = feat_df['Cluster'].map(cluster_labels)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    feat_df['PC1'], feat_df['PC2'] = coords[:, 0], coords[:, 1]

    fig = px.scatter(feat_df.reset_index(), x='PC1', y='PC2', color='Cluster_Label',
                      text='Sub-Category', size='Total_Sales_Volume', size_max=40,
                      title='Product Sub-Category Clusters (PCA-reduced)')
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Sub-categories by cluster")
    strategy_map = {
        'High Volume, Stable Demand': 'Maintain steady safety stock; simple reorder-point replenishment.',
        'High Volume, Mature Demand (Core Revenue Driver)': 'Keep core stock high with a solid safety buffer; review monthly.',
        'High-Growth, Volatile Niche Demand': 'Track monthly; avoid long fixed orders but be ready to scale supply up.',
        'Low Volume, High Volatility': 'Keep lean stock with a wide safety buffer; consider made-to-order.',
        'Growing Demand': 'Increase stock allocation proactively; watch for stock-outs.',
        'Declining Demand': 'Reduce future stock commitments; consider clearance.',
        'Moderate, Balanced Demand': 'Standard periodic review; monitor quarterly.',
    }
    display_df = feat_df[['Cluster_Label', 'Total_Sales_Volume', 'YoY_Growth_Rate_pct', 'Sales_Volatility']].copy()
    display_df['Recommended Strategy'] = display_df['Cluster_Label'].map(strategy_map)
    display_df = display_df.reset_index().rename(columns={
        'Total_Sales_Volume': 'Total Sales ($)', 'YoY_Growth_Rate_pct': 'YoY Growth (%)',
        'Sales_Volatility': 'Volatility'})
    display_df['Total Sales ($)'] = display_df['Total Sales ($)'].round(0)
    display_df['YoY Growth (%)'] = display_df['YoY Growth (%)'].round(1)
    display_df['Volatility'] = display_df['Volatility'].round(0)
    st.dataframe(display_df.sort_values('Cluster_Label'), use_container_width=True, hide_index=True)
