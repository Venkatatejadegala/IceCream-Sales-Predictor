"""
Ice Cream Sales Prediction - Web Application
B.TECH - IT - 3 Year - A Section
Subject: Machine Learning (22IT307)
Module2-T1 - Part A Question 7

Web-based application for predicting ice cream sales using polynomial regression.
Opens automatically in Chrome browser.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Ice Cream Sales Predictor",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Ice Cream Sales Predictor - Polynomial Regression Model"
    }
)

# Custom CSS - Modern Dark/Light Theme with Professional Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
    }
    
    /* Prediction Box - Premium Design */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2.5rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4),
                    0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        animation: slideInUp 0.6s ease-out;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .prediction-value {
        font-size: 4rem;
        font-weight: 800;
        margin: 1.5rem 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        letter-spacing: -2px;
    }
    
    /* Metric Cards - Glass Morphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 30px;
        font-size: 0.95rem;
        font-weight: 600;
        margin-left: 0.8rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-normal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .status-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        padding: 1.8rem;
        border-radius: 15px;
        border-left: 5px solid #6366f1;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
    }
    
    /* Temperature Input Section */
    .temp-section {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Custom Slider */
    .stSlider>div>div>div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 10px;
    }
    
    .stSlider>div>div>div>div {
        background: white;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4);
    }
    
    /* Number Input */
    .stNumberInput>div>div>input {
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    /* Radio Buttons */
    .stRadio>div {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    /* Section Headers */
    h3 {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Badge Styles */
    .sales-badge {
        display: inline-block;
        padding: 0.6rem 1.5rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


class IceCreamSalesPredictor:
    """Class to handle polynomial regression model for ice cream sales prediction."""
    
    def __init__(self, csv_file='Temperature_vs_IceCreamSales.csv'):
        """Initialize the predictor with data from CSV file."""
        self.csv_file = csv_file
        self.df = None
        self.temperature = None
        self.sales = None
        self.linear_model = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=2)
        self.poly_model = LinearRegression()
        self.metrics = {}
        
        # Load and prepare data
        self.load_data()
        self.train_models()
        self.evaluate_models()
    
    def load_data(self):
        """Load data from CSV file."""
        csv_path = Path(__file__).parent / self.csv_file
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        # Extract temperature and sales columns
        temp_col = None
        sales_col = None
        
        for col in self.df.columns:
            col_lower = col.lower().strip()
            if 'temperature' in col_lower or 'temp' in col_lower:
                temp_col = col
            if 'sales' in col_lower or 'icecream' in col_lower or 'ice cream' in col_lower:
                sales_col = col
        
        if temp_col is None or sales_col is None:
            raise ValueError(
                f"Required columns not found in CSV file.\n"
                f"Found columns: {list(self.df.columns)}"
            )
        
        self.temperature = self.df[temp_col].values.reshape(-1, 1)
        self.sales = self.df[sales_col].values
    
    def train_models(self):
        """Train both linear and polynomial regression models."""
        # Train linear regression model
        self.linear_model.fit(self.temperature, self.sales)
        
        # Transform features for polynomial regression (degree 2)
        X_poly = self.poly_features.fit_transform(self.temperature)
        self.poly_model.fit(X_poly, self.sales)
    
    def evaluate_models(self):
        """Evaluate both models and calculate performance metrics."""
        # Linear model predictions
        y_pred_linear = self.linear_model.predict(self.temperature)
        
        # Polynomial model predictions
        X_poly = self.poly_features.transform(self.temperature)
        y_pred_poly = self.poly_model.predict(X_poly)
        
        # Calculate metrics
        mse_linear = mean_squared_error(self.sales, y_pred_linear)
        mse_poly = mean_squared_error(self.sales, y_pred_poly)
        r2_linear = r2_score(self.sales, y_pred_linear)
        r2_poly = r2_score(self.sales, y_pred_poly)
        
        self.metrics = {
            'linear': {
                'mse': mse_linear,
                'r2': r2_linear,
                'predictions': y_pred_linear
            },
            'poly': {
                'mse': mse_poly,
                'r2': r2_poly,
                'predictions': y_pred_poly
            }
        }
        
        return self.metrics
    
    def predict(self, temperature, model_type='poly'):
        """Predict sales for a given temperature."""
        temp_array = np.array([[temperature]])
        
        if model_type == 'linear':
            return self.linear_model.predict(temp_array)[0]
        else:
            temp_poly = self.poly_features.transform(temp_array)
            return self.poly_model.predict(temp_poly)[0]
    
    def get_temperature_status(self, temp):
        """Get status indicator for temperature."""
        if temp < 20:
            return "Low", "status-normal"
        elif temp < 30:
            return "Normal", "status-normal"
        elif temp < 35:
            return "High", "status-warning"
        else:
            return "Very High", "status-high"
    
    def get_sales_category(self, sales):
        """Get category for predicted sales."""
        if sales < 300:
            return "Low Sales", "#10b981"
        elif sales < 700:
            return "Moderate Sales", "#f59e0b"
        else:
            return "High Sales", "#ef4444"


@st.cache_data
def load_predictor():
    """Load predictor model (cached for performance)."""
    return IceCreamSalesPredictor()


@st.cache_data
def get_base_plot(predictor):
    """Cache the base regression plot for better performance."""
    return plot_regression(predictor, highlight_temp=None, highlight_pred=None)


def plot_regression(predictor, highlight_temp=None, highlight_pred=None):
    """Create regression plot with both models and highlight prediction."""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Generate points for smooth curves
    X_plot = np.linspace(
        predictor.temperature.min() - 2,
        predictor.temperature.max() + 2,
        300
    ).reshape(-1, 1)
    
    # Scatter plot of actual data
    ax.scatter(
        predictor.temperature.flatten(),
        predictor.sales,
        color='#3b82f6',
        s=250,
        alpha=0.85,
        label='Actual Data Points',
        zorder=5,
        edgecolors='white',
        linewidth=3
    )
    
    # Plot linear regression
    y_linear = predictor.linear_model.predict(X_plot)
    ax.plot(
        X_plot, y_linear,
        color='#ef4444',
        linewidth=3.5,
        label='Linear Regression',
        alpha=0.75,
        linestyle='-',
        zorder=2
    )
    
    # Plot polynomial regression
    X_plot_poly = predictor.poly_features.transform(X_plot)
    y_poly = predictor.poly_model.predict(X_plot_poly)
    ax.plot(
        X_plot, y_poly,
        color='#10b981',
        linewidth=4,
        label='Polynomial Regression (degree=2)',
        alpha=0.85,
        linestyle='--',
        zorder=3
    )
    
    # Highlight prediction point if provided
    if highlight_temp is not None and highlight_pred is not None:
        ax.scatter(
            highlight_temp,
            highlight_pred,
            color='#f59e0b',
            s=600,
            marker='*',
            edgecolors='white',
            linewidth=3,
            label='Your Prediction',
            zorder=10
        )
        # Add annotation
        ax.annotate(
            f'₹{highlight_pred:,.0f}',
            xy=(highlight_temp, highlight_pred),
            xytext=(highlight_temp + 2, highlight_pred + 100),
            fontsize=14,
            fontweight='bold',
            color='#f59e0b',
            arrowprops=dict(
                arrowstyle='->',
                color='#f59e0b',
                lw=2.5,
                connectionstyle='arc3,rad=0.3'
            ),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#f59e0b', lw=2)
        )
        # Reference lines
        ax.axvline(x=highlight_temp, color='#f59e0b', linestyle=':', alpha=0.4, linewidth=2)
        ax.axhline(y=highlight_pred, color='#f59e0b', linestyle=':', alpha=0.4, linewidth=2)
    
    # Add labels and title
    ax.set_title(
        'Ice Cream Sales Prediction Model',
        fontsize=20,
        fontweight='bold',
        pad=25,
        color='#1e293b'
    )
    ax.set_xlabel('Temperature (°C)', fontsize=15, fontweight='600', color='#475569')
    ax.set_ylabel('Ice Cream Sales (₹)', fontsize=15, fontweight='600', color='#475569')
    ax.legend(
        loc='upper left',
        fontsize=12,
        framealpha=0.95,
        shadow=True,
        fancybox=True,
        frameon=True
    )
    ax.grid(True, linestyle='--', alpha=0.4, color='#94a3b8', linewidth=1)
    
    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Set background
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics):
    """Create a bar plot comparing model metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Modern color palette
    colors = ['#ef4444', '#10b981']
    
    # MSE comparison
    bars1 = ax1.bar(
        ['Linear', 'Polynomial'],
        [metrics['linear']['mse'], metrics['poly']['mse']],
        color=colors,
        width=0.7,
        edgecolor='white',
        linewidth=3,
        alpha=0.9
    )
    ax1.set_title(
        'Mean Squared Error (MSE)',
        fontsize=16,
        fontweight='bold',
        pad=20,
        color='#1e293b'
    )
    ax1.set_ylabel('MSE (lower is better)', fontsize=13, fontweight='600', color='#475569')
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y', color='#94a3b8')
    ax1.set_facecolor('#f8fafc')
    
    # Add value labels
    for i, (bar, v) in enumerate(zip(bars1, [metrics['linear']['mse'], metrics['poly']['mse']])):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height + 500,
            f'{v:,.0f}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=13,
            color=colors[i]
        )
    
    # R² comparison
    bars2 = ax2.bar(
        ['Linear', 'Polynomial'],
        [metrics['linear']['r2'], metrics['poly']['r2']],
        color=colors,
        width=0.7,
        edgecolor='white',
        linewidth=3,
        alpha=0.9
    )
    ax2.set_title(
        'R² Score (Coefficient of Determination)',
        fontsize=16,
        fontweight='bold',
        pad=20,
        color='#1e293b'
    )
    ax2.set_ylabel('R² (higher is better)', fontsize=13, fontweight='600', color='#475569')
    ax2.set_ylim([0, 1.2])
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y', color='#94a3b8')
    ax2.set_facecolor('#f8fafc')
    
    # Add value labels
    for i, (bar, v) in enumerate(zip(bars2, [metrics['linear']['r2'], metrics['poly']['r2']])):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.03,
            f'{v:.4f}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=13,
            color=colors[i]
        )
    
    # Style both subplots
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(colors='#475569', labelsize=12)
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


def main():
    """Main application function."""
    # Header with gradient
    st.markdown('<h1 class="main-header">🍦 Ice Cream Sales Predictor</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Advanced Polynomial Regression Model for Sales Prediction</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Load predictor (cached)
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"❌ **Error loading data:** {str(e)}")
        st.info("💡 Please ensure the CSV file 'Temperature_vs_IceCreamSales.csv' exists in the project directory.")
        st.stop()
    
    # Main layout
    col1, col2 = st.columns([1, 1.6])
    
    with col1:
        st.markdown("### 📊 Temperature Input")
        
        # Temperature slider - make it more prominent
        temperature = st.slider(
            "**Temperature (°C):**",
            min_value=float(predictor.temperature.min() - 5),
            max_value=float(predictor.temperature.max() + 5),
            value=25.0,
            step=0.5,
            key="temp_slider",
            help="Adjust the slider to set the temperature"
        )
        
        # Number input as alternative
        st.markdown("**Or enter manually:**")
        temperature_input = st.number_input(
            "Temperature value:",
            min_value=float(predictor.temperature.min() - 5),
            max_value=float(predictor.temperature.max() + 5),
            value=25.0,
            step=0.5,
            key="temp_input",
            label_visibility="collapsed"
        )
        
        # Use the number input if it's different from slider
        if abs(temperature_input - temperature) > 0.01:
            temperature = temperature_input
        
        # Get temperature status
        status_text, status_class = predictor.get_temperature_status(temperature)
        
        # Get color based on status
        if status_class == "status-normal":
            status_color = "#10b981"  # Green
        elif status_class == "status-warning":
            status_color = "#f59e0b"  # Orange
        else:
            status_color = "#ef4444"  # Red
        
        # Display temperature status (just colored text, no background)
        st.markdown(f"""
        <div style="padding: 1rem 0; margin: 1rem 0;">
            <strong style="font-size: 1.05rem; color: #1e293b;">Temperature Status:</strong> 
            <span style="color: {status_color}; font-weight: 600; font-size: 1.05rem;">{status_text}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection
        st.markdown("### 🎯 Model Selection")
        model_type = st.radio(
            "**Choose Regression Model:**",
            ("Polynomial Regression (Recommended)", "Linear Regression"),
            index=0,
            key="model_selection"
        )
        
        model_key = 'poly' if "Polynomial" in model_type else 'linear'
        
        # Predict button
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button(
            "🔮 Predict Sales",
            use_container_width=True,
            type="primary",
            key="predict_button"
        )
        
        # Initialize session state for prediction
        if 'prediction' not in st.session_state:
            st.session_state.prediction = None
            st.session_state.temp_used = None
            st.session_state.model_used = None
        
        # Show prediction only after button click
        if predict_button or st.session_state.prediction is not None:
            # Update prediction when button is clicked or temperature/model changes
            if predict_button or (st.session_state.temp_used != temperature) or (st.session_state.model_used != model_key):
                prediction = predictor.predict(temperature, model_key)
                st.session_state.prediction = prediction
                st.session_state.temp_used = temperature
                st.session_state.model_used = model_key
            else:
                prediction = st.session_state.prediction
            
            sales_category, category_color = predictor.get_sales_category(prediction)
            
            # Display prediction with premium design
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 1.4rem; margin-bottom: 1rem; opacity: 0.95; font-weight: 600; position: relative; z-index: 1;">
                    Predicted Sales
                </div>
                <div class="prediction-value">₹{prediction:,.2f}</div>
                <div style="font-size: 1.15rem; opacity: 0.9; margin-top: 0.8rem; position: relative; z-index: 1;">
                    {model_type} at {temperature}°C
                </div>
                <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 2px solid rgba(255,255,255,0.3); position: relative; z-index: 1;">
                    <span class="sales-badge" style="background: {category_color}; color: white;">
                        {sales_category}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show placeholder when no prediction yet
            st.markdown("""
            <div style="padding: 3rem 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                        border-radius: 20px; margin: 2rem 0; 
                        box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 2px dashed #cbd5e1; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔮</div>
                <div style="font-size: 1.2rem; color: #64748b; font-weight: 600;">
                    Click "Predict Sales" to see the prediction
                </div>
            </div>
            """, unsafe_allow_html=True)
            prediction = None
        
        # Model Performance Metrics
        st.markdown("### 📈 Model Performance")
        
        metrics = predictor.metrics
        
        # Linear Regression metrics
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #1e293b; margin-bottom: 1rem; font-size: 1.3rem; font-weight: 700;">📉 Linear Regression</h4>
            <p style="font-size: 1.1rem; margin: 0.8rem 0; color: #475569;">
                <strong style="color: #1e293b;">MSE:</strong> 
                <span style="color: #ef4444; font-weight: 700;">{metrics['linear']['mse']:,.2f}</span>
            </p>
            <p style="font-size: 1.1rem; margin: 0.8rem 0; color: #475569;">
                <strong style="color: #1e293b;">R² Score:</strong> 
                <span style="color: #10b981; font-weight: 700;">{metrics['linear']['r2']:.4f}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Polynomial Regression metrics
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #1e293b; margin-bottom: 1rem; font-size: 1.3rem; font-weight: 700;">📊 Polynomial Regression</h4>
            <p style="font-size: 1.1rem; margin: 0.8rem 0; color: #475569;">
                <strong style="color: #1e293b;">MSE:</strong> 
                <span style="color: #10b981; font-weight: 700;">{metrics['poly']['mse']:,.2f}</span>
            </p>
            <p style="font-size: 1.1rem; margin: 0.8rem 0; color: #475569;">
                <strong style="color: #1e293b;">R² Score:</strong> 
                <span style="color: #10b981; font-weight: 700;">{metrics['poly']['r2']:.4f}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong style="color: #1e293b; font-size: 1.05rem;">💡 Performance Insight:</strong><br>
            <span style="color: #475569;">Lower MSE and higher R² values indicate better model performance. 
            Polynomial regression demonstrates superior accuracy for this dataset.</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Regression Analysis")
        
        # Regression plot with highlighted prediction (only if prediction exists)
        if st.session_state.get('prediction') is not None:
            fig = plot_regression(
                predictor,
                highlight_temp=st.session_state.temp_used,
                highlight_pred=st.session_state.prediction
            )
        else:
            fig = get_base_plot(predictor)
        st.pyplot(fig, use_container_width=True)
        
        # Metrics comparison plot
        st.markdown("### 📉 Model Comparison")
        fig2 = plot_metrics_comparison(metrics)
        st.pyplot(fig2, use_container_width=True)
        
        # Dataset table
        st.markdown("### 📋 Dataset")
        st.dataframe(
            predictor.df,
            column_config={
                "Temperature": st.column_config.NumberColumn(
                    "Temperature (°C)",
                    format="%.1f °C",
                    width="medium"
                ),
                "IceCream Sales (in ₹)": st.column_config.NumberColumn(
                    "Sales (₹)",
                    format="₹%.2f",
                    width="medium"
                ),
            },
            hide_index=True,
            use_container_width=True,
            height=320
        )
    
    # Sidebar with additional information
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Ice Cream Sales Predictor**
        
        A sophisticated web application using polynomial regression to predict ice cream sales based on temperature.
        
        **Key Features:**
        - 🎯 Real-time predictions
        - 📊 Interactive visualizations
        - 📈 Model performance comparison
        - 🎨 Modern, professional design
        
        **Model Information:**
        - **Linear Regression:** Straight line fit
        - **Polynomial Regression:** Curved line fit (degree 2)
        
        **Performance Metrics:**
        - **MSE:** Mean Squared Error (lower is better)
        - **R²:** Coefficient of Determination (higher is better)
        """)
        
        st.markdown("---")
        st.markdown("### 📚 Model Explanation")
        
        with st.expander("🔍 What is Polynomial Regression?"):
            st.markdown("""
            **Polynomial Regression** is an advanced form of regression analysis where the relationship 
            between the independent variable (temperature) and dependent variable (sales) 
            is modeled as an nth degree polynomial.
            
            **Key Differences:**
            - **Linear:** y = mx + b (straight line)
            - **Polynomial:** y = a₀ + a₁x + a₂x² + ... (curved line)
            
            Polynomial regression excels at capturing non-linear relationships, making it ideal 
            for data with curved patterns like temperature-sales relationships.
            """)
        
        with st.expander("🎯 Why Polynomial Regression?"):
            st.markdown("""
            **Performance Comparison:**
            - Polynomial regression has **lower MSE** (544.64 vs 8,588.01)
            - Polynomial regression has **higher R²** (0.9967 vs 0.9485)
            
            **Conclusion:**
            Polynomial regression is more suitable for this dataset as it better captures 
            the curved relationship between temperature and sales, resulting in significantly 
            more accurate predictions.
            """)
        
        with st.expander("📖 How to Use"):
            st.markdown("""
            1. **Adjust Temperature:** Use the slider or number input
            2. **Select Model:** Choose between Linear or Polynomial regression
            3. **View Prediction:** Sales prediction updates automatically in real-time
            4. **Analyze Results:** Check the visualization and performance metrics
            
            The prediction updates instantly as you change the temperature!
            """)


if __name__ == "__main__":
    main()
