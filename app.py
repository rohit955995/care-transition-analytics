from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Care Transition Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
DATE_COL = "Date"
APPREHENDED_COL = "Children apprehended and placed in CBP custody"
CBP_COL = "Children in CBP custody"
TRANSFER_COL = "Children transferred out of CBP custody"
HHS_COL = "Children in HHS Care"
DISCHARGE_COL = "Children discharged from HHS Care"

REQUIRED_COLS = [
    DATE_COL,
    APPREHENDED_COL,
    CBP_COL,
    TRANSFER_COL,
    HHS_COL,
    DISCHARGE_COL,
]

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def normalize_name(name: str) -> str:
    """Normalize a column name so small CSV naming differences do not break the app."""
    return (
        str(name)
        .replace("*", "")
        .replace("\n", " ")
        .strip()
        .lower()
    )


def find_dataset() -> Path | None:
    """Find the dataset whether it has the original or renamed filename."""
    candidates = [
        Path("data/HHS_Unaccompanied_Alien_Children_Program.csv"),
        Path("data/HHS_Unaccompanied_Alien_Children_Program(2).csv"),
        Path("HHS_Unaccompanied_Alien_Children_Program.csv"),
        Path("HHS_Unaccompanied_Alien_Children_Program(2).csv"),
    ]

    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path

    data_dir = Path("data")
    if data_dir.exists():
        csv_files = sorted(data_dir.glob("*.csv"))
        for path in csv_files:
            if path.stat().st_size > 0:
                return path

    return None


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else 0.0


def safe_percent(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Row-wise percentage with zero-denominator protection."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    den = den.replace(0, np.nan)
    result = num.div(den).mul(100)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def clean_and_standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map dataset column variations to the names used by the application."""
    normalized = {normalize_name(col): col for col in df.columns}

    aliases = {
        DATE_COL: ["date"],
        APPREHENDED_COL: [
            "children apprehended and placed in cbp custody",
            "children apprehended and placed in cbp custody*",
        ],
        CBP_COL: ["children in cbp custody"],
        TRANSFER_COL: ["children transferred out of cbp custody"],
        HHS_COL: ["children in hhs care"],
        DISCHARGE_COL: ["children discharged from hhs care"],
    }

    rename_map = {}
    for standard_name, possible_names in aliases.items():
        for possible in possible_names:
            key = normalize_name(possible)
            if key in normalized:
                rename_map[normalized[key]] = standard_name
                break

    df = df.rename(columns=rename_map).copy()

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    # Date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # Numeric columns: remove commas and convert safely.
    for col in REQUIRED_COLS[1:]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[DATE_COL]).copy()

    # Missing numeric values use 0 for this operational dashboard.
    for col in REQUIRED_COLS[1:]:
        df[col] = df[col].fillna(0)

    # Remove exact duplicate rows and sort chronologically.
    df = df.drop_duplicates().sort_values(DATE_COL).reset_index(drop=True)

    # Derived analytics columns. These are SAFE: no divide-by-zero.
    df["Transfer Efficiency (%)"] = safe_percent(
        df[TRANSFER_COL], df[CBP_COL]
    )
    df["Discharge Effectiveness (%)"] = safe_percent(
        df[DISCHARGE_COL], df[HHS_COL]
    )
    df["CBP Backlog"] = df[CBP_COL]
    df["HHS Backlog"] = df[HHS_COL]
    df["Pipeline Throughput"] = df[TRANSFER_COL] + df[DISCHARGE_COL]
    df["Total Active Care Load"] = df[CBP_COL] + df[HHS_COL]

    # Remove any unexpected infinities from all numeric columns.
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


@st.cache_data(show_spinner="Loading and cleaning HHS data...")
def load_data() -> tuple[pd.DataFrame, str]:
    path = find_dataset()
    if path is None:
        raise FileNotFoundError(
            "CSV dataset not found. Put the HHS CSV inside the data/ folder."
        )

    raw = pd.read_csv(path)
    df = clean_and_standardize_columns(raw)
    return df, str(path)


@st.cache_resource(show_spinner="Training Random Forest model...")
def train_model(df: pd.DataFrame):
    model_df = df[
        [DATE_COL, APPREHENDED_COL, CBP_COL, TRANSFER_COL, HHS_COL]
    ].copy()

    # Convert date into a numeric feature.
    model_df["Date Ordinal"] = model_df[DATE_COL].map(pd.Timestamp.toordinal)

    feature_cols = [
        "Date Ordinal",
        APPREHENDED_COL,
        CBP_COL,
        TRANSFER_COL,
    ]

    X = model_df[feature_cols]
    y = model_df[HHS_COL]

    # Chronological split is more appropriate for historical operational data.
    split_index = max(int(len(model_df) * 0.80), 1)
    if split_index >= len(model_df):
        split_index = len(model_df) - 1

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions) if len(y_test) > 1 else 0.0

    importance = pd.DataFrame(
        {
            "Feature": [
                "Date",
                "Apprehended",
                "CBP Custody",
                "Transferred",
            ],
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    return model, feature_cols, mae, rmse, r2, importance


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
try:
    df, dataset_path = load_data()
except Exception as exc:
    st.error("Unable to load the dataset.")
    st.code(str(exc))
    st.info(
        "Make sure your CSV is inside data/ and contains Date, CBP custody, "
        "transfers, HHS care, apprehensions and discharges columns."
    )
    st.stop()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("🎛️ Dashboard Controls")

min_date = df[DATE_COL].min().date()
max_date = df[DATE_COL].max().date()

selected_range = st.sidebar.date_input(
    "📅 Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date = end_date = selected_range

filtered_df = df[
    (df[DATE_COL].dt.date >= start_date)
    & (df[DATE_COL].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No records are available for the selected date range.")
    st.stop()

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🏥 Care Transition Efficiency & Placement Outcome Analytics")
st.caption(
    "Analytics dashboard for monitoring the care transition pipeline "
    "from CBP custody to HHS care and final placement."
)

# ---------------------------------------------------------
# KPI CALCULATIONS — SAFE, NO INF%
# ---------------------------------------------------------
transfer_efficiency = safe_mean(filtered_df["Transfer Efficiency (%)"])
discharge_effectiveness = safe_mean(
    filtered_df["Discharge Effectiveness (%)"]
)
avg_cbp = safe_mean(filtered_df["CBP Backlog"])
avg_hhs = safe_mean(filtered_df["HHS Backlog"])
avg_throughput = safe_mean(filtered_df["Pipeline Throughput"])
avg_active_load = safe_mean(filtered_df["Total Active Care Load"])

# ---------------------------------------------------------
# KPI DISPLAY
# ---------------------------------------------------------
st.markdown("## 📊 Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🔄 Transfer Efficiency", f"{transfer_efficiency:.2f}%")
with c2:
    st.metric("🏠 Discharge Effectiveness", f"{discharge_effectiveness:.2f}%")
with c3:
    st.metric("👮 CBP Backlog", f"{avg_cbp:,.0f}")
with c4:
    st.metric("🏥 HHS Care", f"{avg_hhs:,.0f}")

c5, c6, c7 = st.columns(3)
with c5:
    st.metric("🚚 Avg Pipeline Throughput", f"{avg_throughput:,.0f}")
with c6:
    st.metric("👥 Active Care Load", f"{avg_active_load:,.0f}")
with c7:
    st.metric("📋 Data Records", f"{len(filtered_df):,}")

# ---------------------------------------------------------
# CARE TRANSITION PIPELINE
# ---------------------------------------------------------
st.markdown("---")
st.markdown("## 🔄 Care Transition Pipeline")

pipeline = pd.DataFrame(
    {
        "Stage": [
            "CBP Custody",
            "Transferred",
            "HHS Care",
            "Discharged",
        ],
        "Average Children": [
            safe_mean(filtered_df[CBP_COL]),
            safe_mean(filtered_df[TRANSFER_COL]),
            safe_mean(filtered_df[HHS_COL]),
            safe_mean(filtered_df[DISCHARGE_COL]),
        ],
    }
)

fig_pipeline = px.bar(
    pipeline,
    x="Stage",
    y="Average Children",
    title="Average Children by Care Transition Stage",
    text_auto=".0f",
)
st.plotly_chart(fig_pipeline, use_container_width=True)

# ---------------------------------------------------------
# TREND ANALYSIS
# ---------------------------------------------------------
st.markdown("## 📈 Care Transition Trends")

trend_cols = [CBP_COL, TRANSFER_COL, HHS_COL, DISCHARGE_COL]
trend_df = filtered_df[[DATE_COL] + trend_cols].copy()
long_df = trend_df.melt(
    id_vars=DATE_COL,
    value_vars=trend_cols,
    var_name="Metric",
    value_name="Children",
)

fig_trend = px.line(
    long_df,
    x=DATE_COL,
    y="Children",
    color="Metric",
    markers=True,
    title="Care Transition Metrics Over Time",
)
st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------------------------------------------
# EFFICIENCY TRENDS
# ---------------------------------------------------------
st.markdown("## ⚡ Efficiency Analysis")

eff_long = filtered_df[
    [DATE_COL, "Transfer Efficiency (%)", "Discharge Effectiveness (%)"]
].melt(
    id_vars=DATE_COL,
    var_name="Metric",
    value_name="Percentage",
)

fig_eff = px.line(
    eff_long,
    x=DATE_COL,
    y="Percentage",
    color="Metric",
    markers=True,
    title="Transfer and Discharge Efficiency",
)
fig_eff.update_yaxes(title="Percentage (%)")
st.plotly_chart(fig_eff, use_container_width=True)

# ---------------------------------------------------------
# BACKLOG / ACTIVE LOAD
# ---------------------------------------------------------
st.markdown("## 🚨 Backlog & Active Care Load")

load_long = filtered_df[
    [DATE_COL, "CBP Backlog", "HHS Backlog", "Total Active Care Load"]
].melt(
    id_vars=DATE_COL,
    var_name="Metric",
    value_name="Children",
)

fig_load = px.line(
    load_long,
    x=DATE_COL,
    y="Children",
    color="Metric",
    title="CBP, HHS and Total Active Care Load",
)
st.plotly_chart(fig_load, use_container_width=True)

# ---------------------------------------------------------
# MACHINE LEARNING
# ---------------------------------------------------------
st.markdown("---")
st.markdown("## 🤖 Machine Learning — HHS Care Prediction")

try:
    model, feature_cols, mae, rmse, r2, importance = train_model(df)

    st.markdown("### 📊 Model Performance")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("MAE", f"{mae:,.2f}")
    with m2:
        st.metric("RMSE", f"{rmse:,.2f}")
    with m3:
        st.metric("R² Score", f"{r2:.4f}")

    # Latest prediction based on the latest actual operational inputs.
    latest = df.sort_values(DATE_COL).iloc[-1]
    latest_input = pd.DataFrame(
        {
            "Date Ordinal": [latest[DATE_COL].toordinal()],
            APPREHENDED_COL: [latest[APPREHENDED_COL]],
            CBP_COL: [latest[CBP_COL]],
            TRANSFER_COL: [latest[TRANSFER_COL]],
        }
    )
    latest_prediction = float(model.predict(latest_input[feature_cols])[0])

    st.markdown("### 🔮 Latest HHS Care Prediction")
    p1, p2 = st.columns(2)
    with p1:
        st.metric("Predicted HHS Care", f"{latest_prediction:,.0f}")
    with p2:
        st.metric("Actual HHS Care", f"{latest[HHS_COL]:,.0f}")

    st.caption(
        "Prediction is an estimate from the historical Random Forest model; "
        "it is not a guaranteed real-world outcome."
    )

    st.markdown("### 🎯 Feature Importance")
    fig_importance = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance",
        text_auto=".3f",
    )
    st.plotly_chart(fig_importance, use_container_width=True)

except Exception as exc:
    st.error("The machine-learning section could not be completed.")
    st.code(str(exc))

# ---------------------------------------------------------
# DATA PREVIEW / DOWNLOAD
# ---------------------------------------------------------
st.markdown("---")
st.markdown("## 📄 Filtered Data")

with st.expander("View filtered records"):
    display_df = filtered_df.copy()
    display_df[DATE_COL] = display_df[DATE_COL].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Filtered CSV",
    data=csv_data,
    file_name="care_transition_filtered_data.csv",
    mime="text/csv",
)

st.caption(f"Dataset loaded from: {dataset_path}")
