# app.py
# Ice Cream Sales & Profit Predictor - Final Premium Version
# Features:
#  - Uses fixed dataset: Ice Cream Sales and Temperature.csv
#  - Trains separate Linear & Polynomial models for Sales and Profit
#  - Auto-selects best polynomial degree (per target) up to sidebar max_degree
#  - Predicts Sales (units) and Profit (INR) for entered Temperature
#  - Predictions displayed as integers and labelled (units / ₹)
#  - Prediction history (session) with download, delete, clear, delete selected
#  - Optional persistent history written to disk
#  - Save / Load polynomial models (joblib) — both Sales & Profit models saved together
#  - Theme toggle (Light / Dark)
#  - Defensive checks, helpful messages, tidy UI
#
# Expected dataset filename (in same folder as app.py):
#    "Ice Cream Sales and Temperature.csv"
# Expected important columns (any acceptable variant):
#    Temperature , Ice Cream Sales (or Sales) , Profit (INR) (or Profit)
#
# Copy this file into your project folder and run:
#    streamlit run app.py
# or use your run_app.py launcher if you have one.

import os
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

# -----------------------------------------------------------------------------
# File constants
# -----------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "Ice Cream Sales and Temperature.csv"
MODEL_FILE = Path(__file__).parent / "best_models.pkl"  # will store both sales & profit models
PERSISTENT_HISTORY_FILE = Path(__file__).parent / "prediction_history_persistent.csv"
LOGO_FILE = Path(__file__).parent / "WhatsApp Image 2025-11-10 at 18.18.41_a06cd751.jpg"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def get_base64_of_bin_file(bin_file):
    """Converts a binary file to a Base64 string for embedding."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        print(f"Error reading file {bin_file}: {e}")
        return None

@st.cache_data
def load_logo_base64(logo_path: Path):
    """Load and cache the logo image as a Base64 string."""
    logo_base64 = get_base64_of_bin_file(logo_path)
    if logo_base64:
        return f"data:image/png;base64,{logo_base64}"
    return None

def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV safely and raise a helpful error if not possible."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")


def detect_column_by_keywords(df: pd.DataFrame, keywords):
    """Return the first column whose name contains any of the given keywords (case-insensitive)."""
    for c in df.columns:
        lc = c.lower()
        for k in keywords:
            if k.lower() in lc:
                return c
    return None


def detect_temperature_column(df: pd.DataFrame):
    """Detect a temperature-like column name."""
    possible = ["temperature", "temp", "°c", "celsius"]
    c = detect_column_by_keywords(df, possible)
    if c:
        return c
    # fallback: first numeric column
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return num[0] if num else None


def detect_sales_column(df: pd.DataFrame, exclude=None):
    """Detect a sales-like column."""
    possible = ["sales", "sale", "ice", "units"]
    c = detect_column_by_keywords(df, possible)
    if exclude and c == exclude:
        # if detected matches excluded, try alternative heuristic
        for name in df.columns:
            if name != exclude and name.lower().strip() not in ["temperature"]:
                if "sale" in name.lower() or "ice" in name.lower():
                    return name
        return None
    return c


def detect_profit_column(df: pd.DataFrame, exclude=None):
    """Detect a profit-like column."""
    possible = ["profit", "profit (inr)", "profit (rs)", "profit(inr)", "profitinr", "₹"]
    c = detect_column_by_keywords(df, possible)
    if c:
        return c
    # fallback: numeric column not temp or sales (if exists)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude:
        num_cols = [nc for nc in num_cols if nc != exclude]
    if len(num_cols) >= 2:
        return num_cols[-1]  # assume last numeric is profit
    return None


def train_best_polynomial(X: np.ndarray, y: np.ndarray, max_degree: int = 6):
    """Train polynomial regression models degrees 1..max_degree, return best by in-sample R²."""
    best = {"degree": 1, "r2": -np.inf, "poly": None, "model": None}
    for d in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=d, include_bias=True)
        Xp = poly.fit_transform(X)
        model = LinearRegression().fit(Xp, y)
        try:
            r2 = float(r2_score(y, model.predict(Xp)))
        except Exception:
            r2 = -np.inf
        if r2 > best["r2"]:
            best.update({"degree": d, "r2": r2, "poly": poly, "model": model})
    return best


def evaluate_metrics(model, X, y):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y, preds))
    return mse, rmse, r2


def build_plotly_profit(X, y, lin_model, poly_obj, poly_model, highlight_point=None):
    """Plot profit actual vs fitted lines (only profit visualization)."""
    X_plot = np.linspace(X.min() - 2, X.max() + 2, 600).reshape(-1, 1)
    y_lin = lin_model.predict(X_plot)
    y_poly = poly_model.predict(poly_obj.transform(X_plot))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode="markers", name="Actual Profit", marker=dict(size=8, color="#3b82f6")))
    fig.add_trace(go.Scatter(x=X_plot.flatten(), y=y_lin.flatten(), mode="lines", name="Linear Fit", line=dict(color="#ef4444", width=2.6)))
    fig.add_trace(go.Scatter(x=X_plot.flatten(), y=y_poly.flatten(), mode="lines", name="Polynomial Fit", line=dict(color="#10b981", width=2.6, dash="dash")))
    if highlight_point:
        t, p = highlight_point
        fig.add_trace(go.Scatter(x=[t], y=[p], mode="markers+text", text=[f"₹{int(round(p))} profit"], textposition="top center",
                                 name="Prediction", marker=dict(size=14, color="#f59e0b", symbol="star")))
    fig.update_layout(template="simple_white", height=520, margin=dict(l=20, r=20, t=40, b=40))
    fig.update_xaxes(title_text="Temperature (°C)")
    fig.update_yaxes(title_text="Profit (INR)")
    return fig


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ice Cream Sales & Profit Predictor",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Styling (Modified to include logo and university info)
# -----------------------------------------------------------------------------
LOGO_DATA_URI = load_logo_base64(LOGO_FILE)

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }
.header { font-size:2.4rem; font-weight:800;
    background:linear-gradient(135deg,#667eea,#764ba2,#f093fb);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sub { color:#64748b; margin-bottom:1rem; }
.prediction-box {
    border-radius:12px; padding:14px; color:#fff; text-align:center;
    box-shadow:0 12px 40px rgba(102,126,234,0.12);
}
.metric-card {
    background: rgba(255,255,255,0.98);
    border-left: 6px solid #667eea;
    padding:12px; border-radius:10px; box-shadow:0 8px 30px rgba(0,0,0,0.04);
}
.small-muted { color:#6b7280; font-size:0.95rem; }
.footer { text-align:center; color:#6b7280; font-size:0.9rem; padding-top:12px; }
#MainMenu, footer { visibility: hidden; }

/* Custom styles for logo and header */
.university-header {
    text-align: center;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid #e0e0e0;
}
.university-info {
    font-size: 1.1rem;
    color: #333;
    margin-top: 5px;
}
.dept-info {
    font-size: 1.3rem;
    font-weight: 600;
    color: #004d99; /* Darker blue for emphasis */
    margin-bottom: 15px;
}
.logo-image {
    max-width: 250px;
    height: auto;
    margin-bottom: 10px;
}
</style>
"""

DARK_CSS = """
<style>
body { background-color: #071022; color: #e6eef8; }
.metric-card { background: rgba(10,10,20,0.6); border-left: 6px solid #7c3aed; color:#e6eef8; }
.small-muted { color:#94a3b8; }
.sub { color:#94a3b8; }
.footer { color:#94a3b8; }
.university-header { border-bottom: 1px solid #1f2937; }
.university-info { color: #e6eef8; }
.dept-info { color: #667eea; }
</style>
"""

LIGHT_BG = """
<style>
body { background-color: #f8fafc; color: #0f172a; }
</style>
"""

# Header HTML including logo and department info
HEADER_HTML = f"""
    <div class='university-header'>
        <img src='{LOGO_DATA_URI}' class='logo-image'>
        <div class='dept-info'>Department of Information Technology</div>
        <h1 class='header'>🍦 Ice Cream Sales & Profit Predictor</h1>
        <p class='sub'>Predict Sales (units) and Profit (INR) from Temperature — Linear & Polynomial</p>
    </div>
""" if LOGO_DATA_URI else """
    <div style='text-align:center; margin-top:8px;'>
        <h1 class='header'>🍦 Ice Cream Sales & Profit Predictor</h1>
        <p class='sub'>Department of Information Technology</p>
        <p class='sub'>Predict Sales (units) and Profit (INR) from Temperature — Linear & Polynomial</p>
    </div>
"""


# -----------------------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Timestamp", "Temperature", "Model", "Predicted Sales (units)", "Predicted Profit (INR)"
    ])
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "theme_dark" not in st.session_state:
    st.session_state.theme_dark = False # Set default to dark mode for better visibility
if "persistent_history" not in st.session_state:
    st.session_state.persistent_history = False
if "loaded_models" not in st.session_state:
    st.session_state.loaded_models = None

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Model Controls")
    st.markdown("This app trains on the fixed dataset `Ice Cream Sales and Temperature.csv` in the project folder.")
    max_degree = st.slider("Max polynomial degree to test (per target)", min_value=2, max_value=12, value=6, step=1,
                           help="Higher degree increases flexibility but may overfit — choose carefully.")
    st.markdown("---")
    st.markdown("Model persistence")
    save_models_btn = st.button("💾 Save current polynomial models (Sales & Profit)")
    load_models_btn = st.button("📂 Load saved models")
    st.markdown("---")
    st.session_state.persistent_history = st.checkbox("Enable persistent history (save to disk)", value=st.session_state.persistent_history,
                                                     help="When enabled, session history will be saved to disk automatically.")
    st.markdown("---")
    st.markdown("Theme")
    theme_choice = st.radio("Choose theme", ("Light", "Dark"), index=1 if st.session_state.theme_dark else 0)
    st.session_state.theme_dark = True if theme_choice == "Dark" else False
    st.markdown("---")
    st.markdown("Quick actions")
    if st.button("🔄 Reset session (clear history & last prediction)"):
        st.session_state.history = pd.DataFrame(columns=[
            "Timestamp", "Temperature", "Model", "Predicted Sales (units)", "Predicted Profit (INR)"
        ])
        st.session_state.last_prediction = None
        st.success("Session reset complete")

# Apply CSS
st.markdown(BASE_CSS, unsafe_allow_html=True)
if st.session_state.theme_dark:
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_BG, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------
try:
    df = safe_read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Dataset load error: {e}")
    st.stop()

# Show quick dataset info in sidebar
st.sidebar.markdown(f"**Dataset:** `{DATA_FILE.name}`")
st.sidebar.markdown(f"Rows: **{len(df):,}** — Columns: **{len(df.columns)}**")

# -----------------------------------------------------------------------------
# Column detection
# -----------------------------------------------------------------------------
temp_col = detect_temperature_column(df)
sales_col = detect_sales_column(df, exclude=temp_col)
profit_col = detect_profit_column(df, exclude=temp_col)

if temp_col is None:
    st.error("Could not detect a temperature column. Please ensure dataset contains a temperature column (e.g., 'Temperature').")
    st.stop()
if sales_col is None:
    st.error("Could not detect a sales column. Please ensure dataset contains a sales column (e.g., 'Ice Cream Sales').")
    st.stop()
if profit_col is None:
    st.error("Could not detect a profit column. Please ensure dataset contains a profit column (e.g., 'Profit (INR)').")
    st.stop()

# Clean and coerce numeric
df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
df[profit_col] = pd.to_numeric(df[profit_col], errors="coerce")
df = df.dropna(subset=[temp_col, sales_col, profit_col]).reset_index(drop=True)

if df.empty:
    st.error("No valid rows after cleaning. Check CSV values.")
    st.stop()

# Internal standardized names
T_COL = "Temperature_internal"
S_COL = "Sales_internal"
P_COL = "Profit_internal"

df[T_COL] = df[temp_col].astype(float)
df[S_COL] = df[sales_col].astype(float)
df[P_COL] = df[profit_col].astype(float)

# Features and targets
X = df[[T_COL]].values
y_sales = df[S_COL].values
y_profit = df[P_COL].values

# -----------------------------------------------------------------------------
# Train-Test split for quick validation
# -----------------------------------------------------------------------------
try:
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_sales, test_size=0.18, random_state=42)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_profit, test_size=0.18, random_state=42)
except Exception:
    X_train_s, X_test_s, y_train_s, y_test_s = X, X, y_sales, y_sales
    X_train_p, X_test_p, y_train_p, y_test_p = X, X, y_profit, y_profit

# -----------------------------------------------------------------------------
# Train Linear models
# -----------------------------------------------------------------------------
sales_lin_model = LinearRegression().fit(X_train_s, y_train_s)
profit_lin_model = LinearRegression().fit(X_train_p, y_train_p)

# -----------------------------------------------------------------------------
# Train best Polynomial models (independently for sales & profit)
# -----------------------------------------------------------------------------
best_poly_sales = train_best_polynomial(X, y_sales, max_degree=max_degree)
sales_poly_obj = best_poly_sales["poly"]
sales_poly_model = best_poly_sales["model"]

best_poly_profit = train_best_polynomial(X, y_profit, max_degree=max_degree)
profit_poly_obj = best_poly_profit["poly"]
profit_poly_model = best_poly_profit["model"]

# -----------------------------------------------------------------------------
# Evaluate metrics (in-sample) for display
# -----------------------------------------------------------------------------
mse_sales_lin, rmse_sales_lin, r2_sales_lin = evaluate_metrics(sales_lin_model, X, y_sales)
mse_sales_poly, rmse_sales_poly, r2_sales_poly = evaluate_metrics(sales_poly_model, sales_poly_obj.transform(X), y_sales)

mse_profit_lin, rmse_profit_lin, r2_profit_lin = evaluate_metrics(profit_lin_model, X, y_profit)
mse_profit_poly, rmse_profit_poly, r2_profit_poly = evaluate_metrics(profit_poly_model, profit_poly_obj.transform(X), y_profit)

# Cross-validated R2 (polynomial) for both targets
try:
    cv_r2_sales = float(np.mean(cross_val_score(sales_poly_model, sales_poly_obj.transform(X), y_sales, cv=5, scoring="r2")))
except Exception:
    cv_r2_sales = float("nan")
try:
    cv_r2_profit = float(np.mean(cross_val_score(profit_poly_model, profit_poly_obj.transform(X), y_profit, cv=5, scoring="r2")))
except Exception:
    cv_r2_profit = float("nan")

# -----------------------------------------------------------------------------
# Save / Load models handling
# -----------------------------------------------------------------------------
if save_models_btn:
    try:
        saved = {
            "sales": {
                "poly": sales_poly_obj,
                "model": sales_poly_model,
                "degree": best_poly_sales["degree"],
                "trained_rows": len(df)
            },
            "profit": {
                "poly": profit_poly_obj,
                "model": profit_poly_model,
                "degree": best_poly_profit["degree"],
                "trained_rows": len(df)
            },
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        joblib.dump(saved, MODEL_FILE)
        st.sidebar.success(f"Saved models to {MODEL_FILE.name}")
    except Exception as e:
        st.sidebar.error(f"Failed to save models: {e}")

if load_models_btn:
    if MODEL_FILE.exists():
        try:
            loaded = joblib.load(MODEL_FILE)
            st.sidebar.info(f"Loaded models — Sales deg={loaded['sales'].get('degree','?')}, Profit deg={loaded['profit'].get('degree','?')}, saved_at={loaded.get('saved_at','?')}")
            st.session_state.loaded_models = loaded
        except Exception as e:
            st.sidebar.error(f"Failed to load models: {e}")
    else:
        st.sidebar.warning("No saved models found.")

# -----------------------------------------------------------------------------
# Header Display
# -----------------------------------------------------------------------------
st.markdown(HEADER_HTML, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Prediction input form
# -----------------------------------------------------------------------------
st.header("Prediction")
with st.form("predict_form"):
    temp_input = st.number_input("Temperature (°C)", value=30.0, step=0.5, format="%.2f")
    model_choice_display = st.selectbox("Primary model for emphasized display", ("Polynomial (Auto Best)", "Linear"))
    predict_btn = st.form_submit_button("🔮 Predict")

if predict_btn:
    # Sales predictions
    sales_pred_lin = int(round(float(sales_lin_model.predict([[temp_input]])[0])))
    sales_pred_poly = int(round(float(sales_poly_model.predict(sales_poly_obj.transform([[temp_input]]))[0])))

    # Profit predictions
    profit_pred_lin = int(round(float(profit_lin_model.predict([[temp_input]])[0])))
    profit_pred_poly = int(round(float(profit_poly_model.predict(profit_poly_obj.transform([[temp_input]]))[0])))

    # store last prediction
    st.session_state.last_prediction = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": float(temp_input),
        "sales_linear": sales_pred_lin,
        "sales_poly": sales_pred_poly,
        "profit_linear": profit_pred_lin,
        "profit_poly": profit_pred_poly,
        "sales_poly_deg": best_poly_sales["degree"],
        "profit_poly_deg": best_poly_profit["degree"]
    }

    # Display predictions: both models side-by-side with sales & profit
    st.markdown("### Predicted results")
    c1, c2 = st.columns(2)
    with c1:
        # Linear card
        st.markdown(f"""
            <div class='prediction-box' style='background:linear-gradient(135deg,#ef4444,#fb7185);'>
                <h4 style='margin:0'>Linear Model</h4>
                <div style='font-size:1.1rem;font-weight:700;margin-top:8px'>Sales: {sales_pred_lin} units</div>
                <div style='font-size:1.6rem;font-weight:900;margin-top:6px'>₹{profit_pred_lin} profit</div>
                <div class='small-muted'>At {temp_input} °C</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        # Polynomial card
        st.markdown(f"""
            <div class='prediction-box' style='background:linear-gradient(135deg,#10b981,#34d399);'>
                <h4 style='margin:0'>Polynomial Model (Sales deg {best_poly_sales['degree']} | Profit deg {best_poly_profit['degree']})</h4>
                <div style='font-size:1.1rem;font-weight:700;margin-top:8px'>Sales: {sales_pred_poly} units</div>
                <div style='font-size:1.6rem;font-weight:900;margin-top:6px'>₹{profit_pred_poly} profit</div>
                <div class='small-muted'>At {temp_input} °C</div>
            </div>
        """, unsafe_allow_html=True)

    # Append both model rows to history to preserve record (one row per model)
    rows = [
        {"Timestamp": st.session_state.last_prediction["timestamp"], "Temperature": st.session_state.last_prediction["temperature"],
         "Model": "Linear", "Predicted Sales (units)": sales_pred_lin, "Predicted Profit (INR)": profit_pred_lin},
        {"Timestamp": st.session_state.last_prediction["timestamp"], "Temperature": st.session_state.last_prediction["temperature"],
         "Model": f"Polynomial (Sales deg {best_poly_sales['degree']} | Profit deg {best_poly_profit['degree']})",
         "Predicted Sales (units)": sales_pred_poly, "Predicted Profit (INR)": profit_pred_poly}
    ]
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame(rows)], ignore_index=True)

    # persist if enabled
    if st.session_state.persistent_history:
        try:
            st.session_state.history.to_csv(PERSISTENT_HISTORY_FILE, index=False)
            st.success(f"History persisted to {PERSISTENT_HISTORY_FILE.name}")
        except Exception as e:
            st.error(f"Could not persist history: {e}")

else:
    # If not predicting right now, show last prediction summary if available
    last = st.session_state.get("last_prediction")
    if last:
        # Display the result from the model chosen in the select box
        if model_choice_display == "Linear":
            sales_display = last['sales_linear']
            profit_display = last['profit_linear']
            model_info = "Linear"
        else: # Polynomial
            sales_display = last['sales_poly']
            profit_display = last['profit_poly']
            model_info = f"Polynomial (Sales deg {last['sales_poly_deg']} | Profit deg {last['profit_poly_deg']})"
            
        st.markdown("### Last prediction (session)")
        st.markdown(f"<div class='prediction-box'>Sales: **{sales_display}** units · **₹{profit_display}** profit ({model_info}) <br><span class='small-muted'>{last['timestamp']}</span></div>", unsafe_allow_html=True)
    else:
        st.info("Enter temperature (default 30°C) and click Predict to get Sales & Profit estimates.")

# -----------------------------------------------------------------------------
# Visualization: Profit regression only (clean)
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("Profit Regression (Visualization)")

highlight = None
if st.session_state.get("last_prediction"):
    # Determine which prediction point to highlight based on selected model
    if model_choice_display == "Linear":
        highlight = (st.session_state.last_prediction["temperature"], st.session_state.last_prediction["profit_linear"])
    else: # Polynomial (Auto Best)
        highlight = (st.session_state.last_prediction["temperature"], st.session_state.last_prediction["profit_poly"])

profit_fig = build_plotly_profit(X, y_profit, profit_lin_model, profit_poly_obj, profit_poly_model, highlight_point=highlight)
st.plotly_chart(profit_fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Model Performance: show metrics for both targets & models
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------  
# Optimized Model Performance Summary Table  
# -----------------------------------------------------------------------------  
st.markdown("---")
st.header("📊 Model Performance Summary")

# Prepare the data for performance comparison
metrics_data = {
    "Model Type": [
        "Sales - Linear",
        f"Sales - Polynomial (deg {best_poly_sales['degree']})",
        "Profit - Linear",
        f"Profit - Polynomial (deg {best_poly_profit['degree']})"
    ],
    "MSE": [mse_sales_lin, mse_sales_poly, mse_profit_lin, mse_profit_poly],
    "RMSE": [rmse_sales_lin, rmse_sales_poly, rmse_profit_lin, rmse_profit_poly],
    "R²": [r2_sales_lin, r2_sales_poly, r2_profit_lin, r2_profit_poly],
}

metrics_df = pd.DataFrame(metrics_data).round(4)

# Identify best-performing models
best_r2_idx = metrics_df["R²"].idxmax()
best_mse_idx = metrics_df["MSE"].idxmin()
best_rmse_idx = metrics_df["RMSE"].idxmin()

# Highlight function for Streamlit dataframe styling
def highlight_best(row):
    color = "#d1fae5" if row.name in [best_r2_idx, best_mse_idx, best_rmse_idx] else ""
    return ["background-color: {}".format(color) for _ in row]

styled_df = metrics_df.style.apply(highlight_best, axis=1)

# Custom CSS for better presentation
st.markdown("""
    <style>
        .stDataFrame {font-size: 0.95rem;}
        th, td {text-align: center !important;}
        table {border-collapse: collapse; width: 100%;}
        thead tr {background-color: #e0f2fe; font-weight: 600;}
    </style>
""", unsafe_allow_html=True)

# Display optimized styled table
st.dataframe(styled_df, use_container_width=True, height=240)

# Add textual insight
best_model_name = metrics_df.loc[best_r2_idx, "Model Type"]
st.success(f"✅ **Best overall performance (by R²):** {best_model_name}")

# -----------------------------------------------------------------------------
# Prediction history controls & display
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("Prediction History")

h1, h2, h3 = st.columns([1, 1, 1])
with h1:
    if st.button("Delete last entry"):
        if st.session_state.history.empty:
            st.warning("History is empty.")
        else:
            st.session_state.history = st.session_state.history.iloc[:-1].reset_index(drop=True)
            st.success("Deleted last history entry.")
with h2:
    if st.button("Clear history"):
        st.session_state.history = pd.DataFrame(columns=[
            "Timestamp", "Temperature", "Model", "Predicted Sales (units)", "Predicted Profit (INR)"
        ])
        try:
            if PERSISTENT_HISTORY_FILE.exists():
                PERSISTENT_HISTORY_FILE.unlink()
        except Exception:
            pass
        st.success("Cleared history (session and persistent file removed if existed).")
with h3:
    if st.session_state.history.empty:
        st.download_button("Download history (empty)", data=b"", file_name="prediction_history.csv", mime="text/csv", disabled=True)
    else:
        csv_bytes = df_to_csv_bytes(st.session_state.history)
        st.download_button("⬇️ Download history as CSV", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv")

st.markdown("**Manage rows (delete by index):**")
if not st.session_state.history.empty:
    # Safely determine the max index
    max_idx = max(0, len(st.session_state.history)-1)
    
    # Use a try-except block for value access in case the history is briefly manipulated
    try:
        # Check if the existing value is within the new range
        default_val = st.session_state.get("index_to_delete_val", 0)
        if default_val > max_idx:
            default_val = max_idx

        idx_to_delete = st.number_input("Index to delete", min_value=0, max_value=max_idx, value=default_val, step=1, key="index_to_delete_val")
        
        if st.button("Delete selected index"):
            st.session_state.history = st.session_state.history.drop(st.session_state.history.index[idx_to_delete]).reset_index(drop=True)
            st.success(f"Deleted history row at index {idx_to_delete}")
            st.rerun() # Rerun to update max_idx and number_input value

    except Exception as e:
        st.error(f"Failed to delete index: {e}")

if st.session_state.history.empty:
    st.info("No saved predictions yet. Predictions are auto-added to history on Predict.")
else:
    # Display index for easy deletion reference
    display_history = st.session_state.history.copy()
    display_history.index.name = "Index" 
    display_history = display_history.reset_index()

    st.dataframe(display_history, use_container_width=True, height=260)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("<div class='footer'>© 2025 Ice Cream Sales & Profit Predictor</div>", unsafe_allow_html=True)
# End of file