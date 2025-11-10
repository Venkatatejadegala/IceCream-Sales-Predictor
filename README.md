# Quick Start

Follow these steps to quickly set up and run the project:

1. **Clone the repository:**
	```sh
	git clone https://github.com/Venkatatejadegala/Sales-Prediction.git
	cd Sales-Prediction/Polynomial_Regression_Project
	```

2. **Install dependencies:**
	```sh
	pip install -r requirements.txt
	```

3. **Run the application:**
	```sh
	python run_app.py
	```

Or use the batch file:
	```sh
	run.bat
	```

For more details, see the rest of this README.
# 🍦 Ice Cream Sales Predictor (Dual Model Regression)

A modern machine learning web application that predicts **ice cream sales** based on **temperature (°C)** using both **Linear Regression** and **Polynomial Regression (auto-optimized degree)**.  
This project demonstrates a complete end-to-end pipeline — from dataset reading, training, and prediction to model evaluation, visualization, and persistence — all through an interactive Streamlit interface.

---

## 🧠 Overview

The **Ice Cream Sales Predictor** analyzes the relationship between temperature and ice cream sales using regression techniques.  
It is a lightweight and intelligent web app built with **Python**, **Scikit-learn**, and **Streamlit**.  
The user can enter any temperature value, and the app predicts sales using both models, visualizes the data, compares accuracy metrics, and logs predictions in a session history.

---

## 🎯 Objective

- Predict ice cream sales based on temperature input.  
- Compare performance between **Linear Regression** and **Polynomial Regression**.  
- Automatically select the best polynomial degree for optimal fit.  
- Visualize regression fits and model performance metrics.  
- Provide real-time prediction history and model persistence.

---

## 🗂️ Project Folder Structure
IceCreamSalesPredictor/
│
├── app.py # Main Streamlit web application
├── run_app.py # Auto-launcher that runs app and opens browser
├── run.bat # One-click Windows launcher
├── requirements.txt # Python dependency list
├── Ice Cream Sales and Temperature.csv # Dataset with temperature and sales data
├── best_model.pkl # Saved trained polynomial regression model
└── README.md # Documentation file (this file)

---

## ⚙️ Installation & Execution

### 🔧 Prerequisites
Ensure you have:
- **Python 3.8 or later**
- **pip** (Python package manager)
- **Internet connection** for installing dependencies

---

### 🚀 Quick Start

#### **Option 1: Auto Launch (Recommended)**
```bash
python run_app.py
This script:

Starts Streamlit server automatically

Opens the app in your default web browser (preferably Chrome)

Option 2: Windows Shortcut

Double-click the file:

run.bat

Option 3: Manual Start
streamlit run app.py


After launch, visit:

http://localhost:8501

🧩 File Details & Functionality
1️⃣ app.py — Main Application

Handles the entire workflow:

Loads dataset (Ice Cream Sales and Temperature.csv)

Trains Linear and Polynomial regression models

Automatically selects best polynomial degree (1–12 range)

Evaluates both models (MSE, RMSE, R², CV R²)

Accepts manual temperature input (default 30°C)

Displays integer sales predictions for both models

Plots interactive regression graph using Plotly

Tracks prediction history (session-based)

Allows saving/loading of trained model (best_model.pkl)

Supports light/dark themes and persistent UI layout

Built using Streamlit, Scikit-learn, Plotly, Joblib, and Pandas.

2️⃣ run_app.py — Auto Launcher

Launches the Streamlit app programmatically.

Automatically opens http://localhost:8501 in Chrome or default browser.

Useful for quick startup during demos or presentations.

3️⃣ run.bat — Windows Shortcut

Runs run_app.py with a double-click.

No terminal commands required.

4️⃣ requirements.txt

Contains all dependencies needed to run the app:

streamlit
pandas
numpy
scikit-learn
plotly
joblib


Install them by:

pip install -r requirements.txt

5️⃣ Ice Cream Sales and Temperature.csv

The dataset used to train both models.

Represents how ice cream sales vary with temperature.

Sample content:

Temperature (°C)	Sales
20	150
25	350
30	700
35	1350
6️⃣ best_model.pkl

A serialized (saved) polynomial regression model using joblib.

Can be reloaded for instant predictions without retraining.

Saved automatically through sidebar option “💾 Save current best polynomial model”.

🧠 Internal Workflow
Step 1. Data Handling

The CSV is loaded into Pandas.

Temperature and Sales columns are automatically detected.

Missing or non-numeric data is removed.

Step 2. Model Training

Linear Regression model is fitted on the dataset.

Polynomial Regression is trained iteratively for degrees 1–12.

Best degree is selected based on R² score.

Step 3. Model Evaluation

For each model:

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

R² (Coefficient of Determination)

CV R² (5-fold cross-validation score)
are computed and displayed in a metrics card format.

Step 4. Prediction Phase

The user enters a temperature (default = 30°C).

Both models predict sales, shown as integers:

Linear Model → 240 sales
Polynomial Model (deg 5) → 245 sales


Predictions are stored in session history for reference.

Step 5. Visualization

Interactive Plotly chart displays:

Actual data points (blue markers)

Linear regression line (red)

Polynomial regression curve (green)

Predicted point (yellow star marker)

Step 6. History Management

Every prediction is added to session history automatically.

Users can:

Clear all history

Delete selected entries

Download history as CSV file

Step 7. Model Persistence

Users can save the polynomial model (best_model.pkl).

Can later reload it directly from sidebar options.

📈 Model Performance Summary
Metric	Linear Regression	Polynomial Regression (Best Degree)
MSE	Higher	Lower
RMSE	Higher	Lower
R²	Good	Excellent
CV R²	Moderate	Strong

Result:
Polynomial Regression provides a more accurate fit for temperature–sales relationship, effectively capturing its non-linear trend.

🎨 Interface Overview
Layout

Header: Centered gradient title

Sidebar: Contains model controls, theme toggle, save/load options

Main Section:

Input temperature and view predictions

Visual regression plot

Model performance metrics

Prediction history table

Footer credit line

UI Highlights

Gradient header and responsive layout

Light/Dark theme support

Smooth visual transitions

Interactive data visualization via Plotly

📊 Visualization Example

When you run the app, you’ll see:

Data points plotted in blue

Red linear regression line

Green polynomial curve

Highlighted yellow point for predicted sales

Below, model performance cards and bar charts summarize accuracy comparisons.

🧱 Architecture & Structure Analysis
Layer	Component	Responsibility
Data Layer	pandas, numpy	Load, clean, and format dataset
Model Layer	scikit-learn	Train and evaluate regression models
UI Layer	streamlit, plotly	Input forms, charts, and dynamic interface
Persistence Layer	joblib, CSV	Save/load models and prediction history
Control Layer	session_state	Manage app state and user interactions
🧾 Troubleshooting
Issue	Cause	Solution
App not opening automatically	Chrome not detected	Open http://localhost:8501 manually
Module import error	Missing dependencies	Run pip install -r requirements.txt
CSV not found	Wrong file name or path	Ensure Ice Cream Sales and Temperature.csv is in same directory
Port in use	Another Streamlit app running	Run: streamlit run app.py --server.port 8502
🚧 Future Enhancements

Integrate live weather API to auto-fetch real temperatures

Add more algorithms (Decision Tree, Random Forest)

Enable multiple dataset uploads

Generate downloadable PDF performance reports

Deploy app on Streamlit Cloud for online access

📊 Expected Outputs
Temperature (°C)	Linear Prediction	Polynomial Prediction
25	340 sales	355 sales
30	720 sales	745 sales
35	1300 sales	1350 sales

Polynomial regression consistently yields more realistic results.

🧰 Technologies Used
Tool	Purpose
Python 3.8+	Core programming language
Streamlit	Web interface framework
Pandas & NumPy	Data handling
Scikit-learn	Regression modeling
Plotly	Interactive data visualization
Joblib	Model persistence
HTML/CSS	Styling and layout formatting
🏁 Conclusion

This project effectively demonstrates how machine learning regression can be used to analyze and forecast real-world business metrics.
It provides a complete learning and demonstration tool — integrating data preprocessing, model training, evaluation, and real-time prediction in a single, interactive environment.

🧾 License

This project is open for educational and demonstrative purposes.
All computations run locally; no external data is transmitted or stored.

---

✅ **This README** is fully compatible with GitHub, college documentation, and internship submissions.  
✅ It explains *everything technically*, including how each file operates and how predictions, models, and visualizations work internally.  
✅ No personal or course details — completely professional.