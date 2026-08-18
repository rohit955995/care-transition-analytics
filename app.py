import streamlit as st
import plotly.express as px

from analysis import load_data, calculate_kpis
from ml_model import prepare_data, train_model


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Care Transition Analytics",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# DATA PATH
# =========================================================

DATA_PATH = "data/HHS_Unaccompanied_Alien_Children_Program.csv"


# =========================================================
# LOAD DATA
# =========================================================

try:

    df = load_data(DATA_PATH)
    df = calculate_kpis(df)

    ml_df = prepare_data(DATA_PATH)

    model, mae, rmse, r2, feature_importance = train_model(ml_df)

except Exception as e:

    st.error("❌ Error loading project data/model")

    st.code(str(e))

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title(
    "🏥 Care Transition Efficiency & Placement Outcome Analytics"
)

st.write(
    "Analytics dashboard for monitoring the care transition "
    "pipeline from CBP custody to HHS care and final placement."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎛️ Dashboard Controls")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

selected_dates = st.sidebar.date_input(
    "📅 Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)


if len(selected_dates) == 2:

    start_date = selected_dates[0]
    end_date = selected_dates[1]

    filtered_df = df[
        (df["Date"].dt.date >= start_date)
        &
        (df["Date"].dt.date <= end_date)
    ]

else:

    filtered_df = df.copy()


# =========================================================
# KPI CALCULATIONS
# =========================================================

transfer_efficiency = filtered_df[
    "Transfer Efficiency (%)"
].mean()

discharge_effectiveness = filtered_df[
    "Discharge Effectiveness (%)"
].mean()

avg_cbp = filtered_df[
    "CBP Backlog"
].mean()

avg_hhs = filtered_df[
    "HHS Backlog"
].mean()

avg_throughput = filtered_df[
    "Pipeline Throughput"
].mean()

avg_active_load = filtered_df[
    "Total Active Care Load"
].mean()


# =========================================================
# MAIN KPIs
# =========================================================

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🔄 Transfer Efficiency",
        f"{transfer_efficiency:.2f}%"
    )


with col2:

    st.metric(
        "🏠 Discharge Effectiveness",
        f"{discharge_effectiveness:.2f}%"
    )


with col3:

    st.metric(
        "👮 CBP Backlog",
        f"{avg_cbp:,.0f}"
    )


with col4:

    st.metric(
        "🏥 HHS Care",
        f"{avg_hhs:,.0f}"
    )


# =========================================================
# SECOND KPI ROW
# =========================================================

col5, col6, col7 = st.columns(3)


with col5:

    st.metric(
        "🚚 Avg Pipeline Throughput",
        f"{avg_throughput:,.0f}"
    )


with col6:

    st.metric(
        "👥 Active Care Load",
        f"{avg_active_load:,.0f}"
    )


with col7:

    st.metric(
        "📋 Data Records",
        f"{len(filtered_df):,}"
    )


st.divider()


# =========================================================
# CARE TRANSITION PIPELINE
# =========================================================

st.subheader("🔄 Care Transition Pipeline")


pipeline_df = {
    "Stage": [
        "CBP Custody",
        "Transferred to HHS",
        "HHS Care",
        "Discharged"
    ],

    "Children": [
        filtered_df[
            "Children in CBP custody"
        ].mean(),

        filtered_df[
            "Children transferred out of CBP custody"
        ].mean(),

        filtered_df[
            "Children in HHS Care"
        ].mean(),

        filtered_df[
            "Children discharged from HHS Care"
        ].mean()
    ]
}


fig_pipeline = px.bar(
    pipeline_df,
    x="Stage",
    y="Children",
    title="Average Children by Care Transition Stage",
    text_auto=".0f"
)

st.plotly_chart(
    fig_pipeline,
    use_container_width=True
)


# =========================================================
# BACKLOG TREND
# =========================================================

st.subheader("📈 Backlog Monitoring")


fig_backlog = px.line(
    filtered_df,
    x="Date",
    y=[
        "CBP Backlog",
        "HHS Backlog"
    ],
    title="CBP Custody vs HHS Care Load",
    labels={
        "value": "Number of Children",
        "variable": "Care Stage"
    }
)

fig_backlog.update_layout(
    hovermode="x unified"
)

st.plotly_chart(
    fig_backlog,
    use_container_width=True
)


# =========================================================
# EFFICIENCY TREND
# =========================================================

st.subheader("⚡ Transition Efficiency")


fig_efficiency = px.line(
    filtered_df,
    x="Date",
    y=[
        "Transfer Efficiency (%)",
        "Discharge Effectiveness (%)"
    ],
    title="Transfer & Discharge Efficiency",
    labels={
        "value": "Efficiency (%)",
        "variable": "Metric"
    }
)

fig_efficiency.update_layout(
    hovermode="x unified"
)

st.plotly_chart(
    fig_efficiency,
    use_container_width=True
)


# =========================================================
# PIPELINE THROUGHPUT
# =========================================================

st.subheader("🚚 Pipeline Throughput")


fig_throughput = px.bar(
    filtered_df,
    x="Date",
    y="Pipeline Throughput",
    title="Daily Pipeline Throughput"
)

st.plotly_chart(
    fig_throughput,
    use_container_width=True
)


# =========================================================
# AUTOMATED INSIGHTS
# =========================================================

st.subheader("💡 Automated Insights")


highest_cbp = filtered_df[
    "CBP Backlog"
].max()

highest_hhs = filtered_df[
    "HHS Backlog"
].max()

best_transfer = filtered_df[
    "Transfer Efficiency (%)"
].max()

best_discharge = filtered_df[
    "Discharge Effectiveness (%)"
].max()


col1, col2 = st.columns(2)


with col1:

    st.info(
        f"""
        **📌 CBP Insight**

        Maximum CBP backlog:

        **{highest_cbp:,.0f} children**
        """
    )

    st.info(
        f"""
        **🔄 Transfer Insight**

        Highest transfer efficiency:

        **{best_transfer:.2f}%**
        """
    )


with col2:

    st.info(
        f"""
        **🏥 HHS Insight**

        Maximum children in HHS care:

        **{highest_hhs:,.0f}**
        """
    )

    st.info(
        f"""
        **🏠 Discharge Insight**

        Highest discharge effectiveness:

        **{best_discharge:.2f}%**
        """
    )


# =========================================================
# MACHINE LEARNING SECTION
# =========================================================

st.divider()

st.header("🤖 Machine Learning — HHS Care Prediction")


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("📊 Model Performance")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "MAE",
        f"{mae:,.2f}"
    )


with col2:

    st.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )


with col3:

    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )


# =========================================================
# LATEST PREDICTION
# =========================================================

st.subheader("🔮 Latest HHS Care Prediction")


latest = ml_df.iloc[-1]


prediction_features = [[

    latest[
        "Children apprehended and placed in CBP custody"
    ],

    latest[
        "Children in CBP custody"
    ],

    latest[
        "Children transferred out of CBP custody"
    ],

    latest[
        "Children discharged from HHS Care"
    ]

]]


predicted_hhs = model.predict(
    prediction_features
)[0]


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "🤖 Predicted HHS Care",
        f"{predicted_hhs:,.0f}"
    )


with col2:

    st.metric(
        "📌 Actual HHS Care",
        f"{latest['Children in HHS Care']:,.0f}"
    )


# =========================================================
# MODEL INSIGHT
# =========================================================

st.info(
    f"""
    🤖 **Model Insight**

    Based on the latest available care-transition
    indicators, the Random Forest model predicts approximately:

    ### {predicted_hhs:,.0f} children

    in HHS care.

    Model R² Score: **{r2:.3f}**
    """
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("🎯 Feature Importance")


fig_importance = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Factors Influencing HHS Care Prediction",
    text_auto=".3f"
)


fig_importance.update_layout(
    yaxis_title="Input Feature",
    xaxis_title="Importance"
)


st.plotly_chart(
    fig_importance,
    use_container_width=True
)


# =========================================================
# DATASET
# =========================================================

st.divider()

st.subheader("📋 Detailed Dataset")


with st.expander("Click to view dataset"):

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🏥 Care Transition Efficiency & Placement Outcome Analytics | "
    "Python • Pandas • Scikit-learn • Plotly • Streamlit"
)