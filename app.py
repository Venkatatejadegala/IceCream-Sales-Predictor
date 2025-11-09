# app.py
# Premium Ice Cream Sales Predictor - Final version
# Developed by Varshitha
# Features:
#  - Uses fixed dataset: Ice Cream Sales and Temperature.csv
#  - Manual temperature input (default 30.0)
#  - Linear and Polynomial regression (auto best degree)
#  - Predictions displayed as integers and labeled with "sales"
#  - Prediction history (session) with download, delete, clear, delete selected
#  - Optional persistent history written to disk
#  - Save / Load polynomial model (joblib)
#  - Theme toggle (Light / Dark)
#  - Sidebar visible by default (initial_sidebar_state="expanded")
#  - Defensive checks, helpful messages, tidy UI

import os
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
from sklearn.model_selection import cross_val_score

# -----------------------------------------
# Page configuration
# -----------------------------------------
st.set_page_config(
    page_title="Ice Cream Sales Predictor",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------
# Constants and files
# -----------------------------------------
DATA_FILE = Path(__file__).parent / "Ice Cream Sales and Temperature.csv"
MODEL_FILE = Path(__file__).parent / "best_model.pkl"
PERSISTENT_HISTORY_FILE = Path(__file__).parent / "prediction_history_persistent.csv"

# -----------------------------------------
# CSS styling
# -----------------------------------------
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
    box-shadow:0 12px 40px rgba(102,126,234,0.18);
}
.metric-card {
    background: rgba(255,255,255,0.98);
    border-left: 6px solid #667eea;
    padding:12px; border-radius:10px; box-shadow:0 8px 30px rgba(0,0,0,0.06);
}
.small-muted { color:#6b7280; font-size:0.95rem; }
.footer { text-align:center; color:#6b7280; font-size:0.9rem; padding-top:12px; }
#MainMenu, footer { visibility: hidden; }
header { visibility: visible; }
</style>
"""

DARK_CSS = """
<style>
body { background-color: #071022; color: #e6eef8; }
.metric-card { background: rgba(10,10,20,0.6); border-left: 6px solid #7c3aed; color:#e6eef8; }
.small-muted { color:#94a3b8; }
.header-sub { color:#94a3b8; }
.footer { color:#94a3b8; }
</style>
"""

LIGHT_BG = """
<style>
body { background-color: #f8fafc; color: #0f172a; }
</style>
"""

# -----------------------------------------
# Helper functions
# -----------------------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV safely and raise a helpful error if not possible."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

def detect_temp_sales_columns(df: pd.DataFrame):
    """Detect temperature and sales columns heuristically; fallback to numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None, None
    temp_col = None
    sales_col = None
    for c in df.columns:
        lc = c.lower()
        if "temp" in lc and temp_col is None:
            temp_col = c
        if ("sale" in lc or "sales" in lc or "ice" in lc) and sales_col is None:
            sales_col = c
    # fallback:
    if temp_col is None:
        temp_col = numeric_cols[0] if len(numeric_cols) > 0 else None
    if sales_col is None:
        sales_col = numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if len(numeric_cols) > 0 else None)
    return temp_col, sales_col

def train_best_polynomial(X: np.ndarray, y: np.ndarray, max_degree: int = 6):
    """Train polynomial models degrees 1..max_degree, return best by training R²."""
    best = {"degree": 1, "r2": -np.inf, "poly": None, "model": None}
    for d in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=d)
        Xp = poly.fit_transform(X)
        model = LinearRegression().fit(Xp, y)
        r2 = r2_score(y, model.predict(Xp))
        if r2 > best["r2"]:
            best.update({"degree": d, "r2": r2, "poly": poly, "model": model})
    return best

def evaluate_metrics(model, X, y):
    """Return mse, rmse, r2 for given model on X,y."""
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y, preds))
    return mse, rmse, r2

def build_plotly_figure(X, y, lin_model, poly_obj, poly_model, highlight_point=None):
    """Construct Plotly figure with data, linear fit, polynomial fit, and optional highlight."""
    X_plot = np.linspace(X.min() - 2, X.max() + 2, 600).reshape(-1, 1)
    y_lin = lin_model.predict(X_plot)
    y_poly = poly_model.predict(poly_obj.transform(X_plot))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode="markers", name="Actual Data",
                             marker=dict(size=8, color="#3b82f6")))
    fig.add_trace(go.Scatter(x=X_plot.flatten(), y=y_lin.flatten(), mode="lines", name="Linear Fit",
                             line=dict(color="#ef4444", width=2.6)))
    fig.add_trace(go.Scatter(x=X_plot.flatten(), y=y_poly.flatten(), mode="lines", name="Polynomial Fit",
                             line=dict(color="#10b981", width=2.6, dash="dash")))
    if highlight_point:
        t, s = highlight_point
        fig.add_trace(go.Scatter(x=[t], y=[s], mode="markers+text", text=[f"{s} sales"],
                                 textposition="top center", name="Prediction",
                                 marker=dict(size=14, color="#f59e0b", symbol="star")))
    fig.update_layout(template="simple_white", height=520, margin=dict(l=20, r=20, t=40, b=40))
    fig.update_xaxes(title_text="Temperature (°C)")
    fig.update_yaxes(title_text="Sales (units)")
    return fig

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------------------
# Session state initialization
# -----------------------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Timestamp", "Temperature", "Predicted Sales", "Model Type"])
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "theme_dark" not in st.session_state:
    st.session_state.theme_dark = False
if "persistent_history" not in st.session_state:
    st.session_state.persistent_history = False

# -----------------------------------------
# Sidebar controls (no upload)
# -----------------------------------------
with st.sidebar:
    st.header("Model Controls")
    st.markdown("This app uses the fixed dataset `Ice Cream Sales and Temperature.csv` in the project folder.")
    max_degree = st.slider("Max polynomial degree to test", min_value=2, max_value=12, value=6, step=1,
                           help="Larger degree can fit more complex shapes but increases overfitting risk.")
    st.markdown("---")
    st.markdown("Model persistence")
    save_model_btn = st.button("💾 Save current best polynomial model")
    load_model_btn = st.button("📂 Load saved model info")
    st.markdown("---")
    st.session_state.persistent_history = st.checkbox("Enable persistent history (save to disk)", value=st.session_state.persistent_history,
                                                     help="When enabled, session history will be written to 'prediction_history_persistent.csv' automatically.")
    st.markdown("---")
    st.markdown("Theme")
    theme_choice = st.radio("Choose theme", ("Light", "Dark"), index=1 if st.session_state.theme_dark else 0)
    st.session_state.theme_dark = True if theme_choice == "Dark" else False
    st.markdown("---")
    st.markdown("Quick actions")
    if st.button("🔄 Reset session (clear history & last prediction)"):
        st.session_state.history = pd.DataFrame(columns=["Timestamp", "Temperature", "Predicted Sales", "Model Type"])
        st.session_state.last_prediction = None
        st.success("Session reset complete")

# Apply CSS and theme
st.markdown(BASE_CSS, unsafe_allow_html=True)
if st.session_state.theme_dark:
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_BG, unsafe_allow_html=True)

# -----------------------------------------
# Load fixed dataset
# -----------------------------------------
try:
    df = safe_read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Dataset load error: {e}")
    st.stop()

# detect columns
temp_col, sales_col = detect_temp_sales_columns(df)
if temp_col is None or sales_col is None:
    st.error("Could not detect temperature/sales columns in dataset. Check file.")
    st.stop()

# coerce numeric and drop NA
df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
df = df.dropna(subset=[temp_col, sales_col]).reset_index(drop=True)
if df.empty:
    st.error("No valid numeric rows after cleaning dataset.")
    st.stop()

X = df[[temp_col]].values
y = df[sales_col].values

# -----------------------------------------
# Train linear and best polynomial models
# -----------------------------------------
lin_model = LinearRegression().fit(X, y)
best_poly = train_best_polynomial(X, y, max_degree=max_degree)
poly_obj = best_poly["poly"]
poly_model = best_poly["model"]

# Evaluate
mse_lin, rmse_lin, r2_lin = evaluate_metrics(lin_model, X, y)
mse_poly, rmse_poly, r2_poly = evaluate_metrics(poly_model, poly_obj.transform(X), y)
try:
    cv_r2 = float(np.mean(cross_val_score(poly_model, poly_obj.transform(X), y, cv=5, scoring="r2")))
except Exception:
    cv_r2 = float("nan")

# Handle model save/load buttons
if save_model_btn:
    try:
        saved = {
            "poly": poly_obj,
            "model": poly_model,
            "degree": best_poly["degree"],
            "temp_col": temp_col,
            "sales_col": sales_col,
            "trained_rows": len(df),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        joblib.dump(saved, MODEL_FILE)
        st.sidebar.success(f"Model saved to {MODEL_FILE.name}")
    except Exception as e:
        st.sidebar.error(f"Failed to save model: {e}")

if load_model_btn:
    if MODEL_FILE.exists():
        try:
            loaded = joblib.load(MODEL_FILE)
            st.sidebar.info(f"Loaded model: degree={loaded.get('degree','?')}, rows={loaded.get('trained_rows','?')}, saved_at={loaded.get('saved_at','?')}")
            st.session_state.loaded_model = loaded
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
    else:
        st.sidebar.warning("No saved model file found.")

# -----------------------------------------
# Top header and dataset info
# -----------------------------------------
# -----------------------------------------
# Centered header (middle of page)
# -----------------------------------------
st.markdown(
    """
    <div style='text-align:center; margin-top:10px;'>
        <h1 class='header' style='margin-bottom:0;'>🍦 Ice Cream Sales Predictor</h1>
        <p class='sub' style='margin-top:4px;'>Predict ice-cream sales from temperature — Linear vs Polynomial</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------
# Prediction input (explicit form)
# -----------------------------------------
st.header("Prediction")

with st.form("predict_form"):
    # default temperature set to 30.0 as requested
    temp_input = st.number_input("Temperature (°C)", value=30.0, step=0.5, format="%.2f")
    model_choice = st.selectbox("Primary model (for display selection)", ("Polynomial (Auto Best)", "Linear"))
    predict_btn = st.form_submit_button("🔮 Predict Sales")

if predict_btn:
    # compute integer predictions (rounded)
    pred_linear = int(round(float(lin_model.predict([[temp_input]])[0])))
    pred_poly = int(round(float(poly_model.predict(poly_obj.transform([[temp_input]]))[0])))

    # set last prediction in session state
    st.session_state.last_prediction = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": float(temp_input),
        "linear": pred_linear,
        "polynomial": pred_poly,
        "poly_degree": best_poly["degree"]
    }

    # Display both side-by-side and show "sales" wording
    st.markdown("### Predicted results")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
            <div class='prediction-box' style='background:linear-gradient(135deg,#ef4444,#fb7185);'>
                <h4 style='margin:0'>Linear</h4>
                <div style='font-size:2rem;font-weight:700;margin-top:8px'>{pred_linear} sales</div>
                <div class='small-muted'>At {temp_input} °C</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class='prediction-box' style='background:linear-gradient(135deg,#10b981,#34d399);'>
                <h4 style='margin:0'>Polynomial (deg {best_poly['degree']})</h4>
                <div style='font-size:2rem;font-weight:700;margin-top:8px'>{pred_poly} sales</div>
                <div class='small-muted'>At {temp_input} °C</div>
            </div>
        """, unsafe_allow_html=True)

    # Auto-add to session history (best UX)
    new_rows = [
        {"Timestamp": st.session_state.last_prediction["timestamp"],
         "Temperature": st.session_state.last_prediction["temperature"],
         "Predicted Sales": st.session_state.last_prediction["linear"],
         "Model Type": "Linear"},
        {"Timestamp": st.session_state.last_prediction["timestamp"],
         "Temperature": st.session_state.last_prediction["temperature"],
         "Predicted Sales": st.session_state.last_prediction["polynomial"],
         "Model Type": f"Polynomial (deg {best_poly['degree']})"}
    ]
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame(new_rows)], ignore_index=True)

    # if persistent enabled, save
    if st.session_state.persistent_history:
        try:
            st.session_state.history.to_csv(PERSISTENT_HISTORY_FILE, index=False)
            st.success(f"History persisted to {PERSISTENT_HISTORY_FILE.name}")
        except Exception as e:
            st.error(f"Could not persist history to disk: {e}")

else:
    # if not just predicted, optionally show last prediction
    last = st.session_state.get("last_prediction")
    if last:
        st.markdown("### Last prediction (session)")
        st.markdown(f"<div class='prediction-box'>{last['linear']} sales · {last['polynomial']} sales (deg {last['poly_degree']})<br><span class='small-muted'>{last['timestamp']}</span></div>", unsafe_allow_html=True)
    else:
        st.info("Enter temperature (default 30°C) and click Predict to generate results.")

# -----------------------------------------
# Visualization
# -----------------------------------------
st.markdown("---")
st.header("Visualization")

highlight = None
if st.session_state.get("last_prediction"):
    highlight = (st.session_state.last_prediction["temperature"], st.session_state.last_prediction["polynomial"])

fig = build_plotly_figure(X, y, lin_model, poly_obj, poly_model, highlight_point=highlight)
st.plotly_chart(fig, width='stretch')

# -----------------------------------------
# Model performance section
# -----------------------------------------
st.markdown("---")
st.header("Model Performance")

mc1, mc2 = st.columns(2)
with mc1:
    st.markdown(f"<div class='metric-card'><b>Linear Regression</b><br>MSE: {mse_lin:.2f}<br>RMSE: {rmse_lin:.2f}<br>R²: {r2_lin:.4f}</div>", unsafe_allow_html=True)
with mc2:
    st.markdown(f"<div class='metric-card'><b>Polynomial Regression (deg {best_poly['degree']})</b><br>MSE: {mse_poly:.2f}<br>RMSE: {rmse_poly:.2f}<br>R²: {r2_poly:.4f}<br>CV R²: {cv_r2:.4f}</div>", unsafe_allow_html=True)

# Comparison chart
comp_fig = go.Figure(data=[
    go.Bar(name="Linear", x=["MSE", "RMSE", "R²"], y=[mse_lin, rmse_lin, r2_lin], marker_color="#ef4444"),
    go.Bar(name=f"Polynomial (deg {best_poly['degree']})", x=["MSE", "RMSE", "R²"], y=[mse_poly, rmse_poly, r2_poly], marker_color="#10b981")
])
comp_fig.update_layout(barmode="group", template="simple_white", height=360)
st.plotly_chart(comp_fig, width='stretch')

# -----------------------------------------
# Prediction history controls & display
# -----------------------------------------
st.markdown("---")
st.header("Prediction History")

hcol1, hcol2, hcol3 = st.columns([1, 1, 1])
with hcol1:
    if st.button("Delete last entry"):
        if st.session_state.history.empty:
            st.warning("History is empty.")
        else:
            st.session_state.history = st.session_state.history.iloc[:-1].reset_index(drop=True)
            st.success("Deleted last history entry.")
with hcol2:
    if st.button("Clear history"):
        st.session_state.history = pd.DataFrame(columns=["Timestamp", "Temperature", "Predicted Sales", "Model Type"])
        # remove persistent file if present
        try:
            if PERSISTENT_HISTORY_FILE.exists():
                PERSISTENT_HISTORY_FILE.unlink()
        except Exception:
            pass
        st.success("Cleared history (session and persistent file removed if existed).")
with hcol3:
    if st.session_state.history.empty:
        st.download_button("Download history (empty)", data=b"", file_name="prediction_history.csv", mime="text/csv", disabled=True)
    else:
        csv_bytes = df_to_csv_bytes(st.session_state.history)
        st.download_button("⬇️ Download history as CSV", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv")

# optional: allow user to delete selected rows by index
st.markdown("**Manage rows (delete by index):**")
if not st.session_state.history.empty:
    st.write("Enter the index (0-based, shown in leftmost column) of the row to delete. Use the table below to find index.")
    idx_to_delete = st.number_input("Index to delete", min_value=0, max_value=max(0, len(st.session_state.history)-1), value=0, step=1)
    if st.button("Delete selected index"):
        try:
            st.session_state.history = st.session_state.history.drop(st.session_state.history.index[idx_to_delete]).reset_index(drop=True)
            st.success(f"Deleted history row at index {idx_to_delete}")
        except Exception as e:
            st.error(f"Failed to delete index {idx_to_delete}: {e}")

# show table
if st.session_state.history.empty:
    st.info("No saved predictions yet. Predictions are auto-added to history on Predict.")
else:
    display_history = st.session_state.history.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)
    st.dataframe(display_history, width='stretch', height=260)

# -----------------------------------------
# Footer
# -----------------------------------------
st.markdown("---")
st.markdown("<div class='footer'>© 2025 Ice Cream Sales Predictor · Developed by Varshitha</div>", unsafe_allow_html=True)
