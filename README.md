# 🍦 Ice Cream Sales Predictor - Polynomial Regression

## 📋 Project Overview

A sophisticated **web-based machine learning application** that predicts ice cream sales based on temperature using polynomial regression. This project demonstrates advanced regression modeling with a modern, interactive user interface.

**Course:** B.TECH - IT - 3 Year - A Section  
**Subject:** Machine Learning (22IT307)  
**Module:** Module2-T1 - Part A Question 7

---

## 🎯 Project Requirements

### Part A Question 7

**Objective:** Build a regression model to predict ice cream sales based on temperature using the provided dataset.

### Sub-questions & Solutions:

1. **a) What is a polynomial regression model? How does it differ from linear regression?** (Understanding - 2 Marks)
   - ✅ Explained in the application sidebar with detailed comparisons
   - Polynomial regression uses higher-degree terms (x², x³, etc.) to capture non-linear relationships
   - Linear regression uses only first-degree terms (x) for straight-line relationships

2. **b) Fit a polynomial regression model to the given dataset to predict IceCream_Sales based on Temperature.** (Applying - 3 Marks)
   - ✅ Implemented with degree 2 polynomial regression
   - ✅ Model trained and ready for predictions
   - ✅ Interactive prediction interface available

3. **c) Analyze the model's performance using Mean Squared Error (MSE) and R² score.** (Analyzing - 2 Marks)
   - ✅ MSE and R² scores calculated and displayed
   - ✅ Visual comparison charts included
   - ✅ Performance metrics shown for both models

4. **d) Evaluate the accuracy of the model and explain whether polynomial regression is suitable for this dataset.** (Evaluating - 3 Marks)
   - ✅ Model evaluation with detailed analysis
   - ✅ Comparison showing polynomial regression superiority
   - ✅ Explanation provided in the application

---

## 📊 Dataset

The dataset `Temperature_vs_IceCreamSales.csv` contains 7 data points:

| Temperature (°C) | IceCream Sales (in ₹) |
|:-----------------|:---------------------|
| 20               | 150                  |
| 22               | 200                  |
| 25               | 350                  |
| 27               | 500                  |
| 30               | 700                  |
| 32               | 1000                 |
| 35               | 1350                 |

---

## ✨ Features

### 🎨 User Interface
- ✅ **Modern Web Design** - Beautiful gradient theme with professional styling
- ✅ **Responsive Layout** - Works perfectly on all screen sizes
- ✅ **Smooth Animations** - Elegant transitions and hover effects
- ✅ **Glass Morphism** - Modern UI elements with backdrop blur effects
- ✅ **Custom Styling** - Poppins font family and custom color scheme

### 🔧 Functionality
- ✅ **Interactive Temperature Input** - Slider and number input options
- ✅ **Temperature Status Indicator** - Color-coded status (Low/Normal/High/Very High)
- ✅ **Model Selection** - Choose between Polynomial (recommended) or Linear Regression
- ✅ **Predict Button** - Click to generate sales prediction
- ✅ **Real-time Updates** - Prediction updates when temperature or model changes

### 📈 Visualizations
- ✅ **Regression Plot** - Shows both Linear and Polynomial regression curves
- ✅ **Prediction Highlighting** - Your prediction point highlighted on the graph
- ✅ **Model Comparison Charts** - Side-by-side MSE and R² comparison
- ✅ **Dataset Table** - View all data points in a formatted table

### 📊 Analytics
- ✅ **Performance Metrics** - MSE and R² scores for both models
- ✅ **Sales Categories** - Low/Moderate/High sales classification
- ✅ **Model Analysis** - Detailed explanations in the sidebar
- ✅ **Educational Content** - Learn about regression models

### 🚀 Performance
- ✅ **Caching** - Optimized with Streamlit caching for fast loading
- ✅ **Auto-launch** - Automatically opens in Chrome browser
- ✅ **Error Handling** - Comprehensive error messages and validation

---

## 🛠️ Installation

### Prerequisites
- **Python 3.7 or higher**
- **pip** (Python package manager)
- **Git** (for cloning the repository)

### Step-by-Step Installation

#### Option 1: Clone from GitHub (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Venkatatejadegala/Sales-Prediction.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd Sales-Prediction
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import streamlit; print('Streamlit installed successfully!')"
   ```

#### Option 2: Download ZIP

1. **Download the repository:**
   - Go to [https://github.com/Venkatatejadegala/Sales-Prediction](https://github.com/Venkatatejadegala/Sales-Prediction)
   - Click the green "Code" button
   - Select "Download ZIP"
   - Extract the ZIP file

2. **Navigate to the project directory:**
   ```bash
   cd Sales-Prediction
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

#### Quick Setup (One Command)
```bash
git clone https://github.com/Venkatatejadegala/Sales-Prediction.git && cd Sales-Prediction && pip install -r requirements.txt
```

---

## 🚀 Usage

### Running the Application

#### **Option 1: Auto-Launcher (Recommended)**
```bash
python run_app.py
```
- Automatically opens Chrome browser
- Starts Streamlit server
- Opens at `http://localhost:8501`

#### **Option 2: Windows Batch File**
Double-click `run.bat` or run:
```bash
run.bat
```

#### **Option 3: Direct Streamlit Command**
```bash
streamlit run app.py
```
Then manually open `http://localhost:8501` in your browser.

### Using the Application

1. **Enter Temperature:**
   - Use the **slider** to select temperature (15°C to 40°C)
   - Or **type directly** in the number input field
   - View the **temperature status** (colored text indicator)

2. **Select Model:**
   - Choose **"Polynomial Regression (Recommended)"** for best accuracy
   - Or select **"Linear Regression"** for comparison

3. **Generate Prediction:**
   - Click the **"🔮 Predict Sales"** button
   - View the predicted sales amount in the gradient box
   - See the **sales category** (Low/Moderate/High)

4. **Analyze Results:**
   - Check **Model Performance** metrics (MSE and R²)
   - View **Regression Plot** with your prediction highlighted
   - Compare models using **Comparison Charts**
   - Review the **Dataset Table**

5. **Learn More:**
   - Explore the **sidebar** for detailed explanations
   - Read about polynomial vs linear regression
   - Understand model evaluation metrics

---

## 📚 Model Explanation

### Polynomial Regression vs Linear Regression

#### **Linear Regression:**
- **Formula:** `y = mx + b`
- **Type:** Straight line
- **Use Case:** Linear relationships
- **Pros:** Simple, interpretable, fast
- **Cons:** Cannot capture curves

#### **Polynomial Regression:**
- **Formula:** `y = a₀ + a₁x + a₂x² + a₃x³ + ...`
- **Type:** Curved line (degree 2 in this project)
- **Use Case:** Non-linear relationships
- **Pros:** Captures curves, more accurate for complex data
- **Cons:** More complex, can overfit

### Why Polynomial Regression for This Dataset?

**Performance Comparison:**
- **MSE:** Polynomial (544.64) vs Linear (8,588.01) - **94% lower error**
- **R² Score:** Polynomial (0.9967) vs Linear (0.9485) - **5% better fit**

**Conclusion:** Polynomial regression is significantly more suitable as it captures the curved relationship between temperature and sales, resulting in much more accurate predictions.

---

## 📈 Model Evaluation

### Performance Metrics

#### **Mean Squared Error (MSE)**
- Measures average squared difference between predicted and actual values
- **Lower is better**
- Formula: `MSE = (1/n) × Σ(actual - predicted)²`

#### **R² Score (Coefficient of Determination)**
- Measures how well the model explains variance in data
- **Higher is better** (range: 0 to 1)
- Formula: `R² = 1 - (SS_res / SS_tot)`

### Results for This Dataset

| Model | MSE | R² Score | Interpretation |
|:------|:----|:---------|:---------------|
| **Linear Regression** | 8,588.01 | 0.9485 | Good fit, but misses curve |
| **Polynomial Regression** | 544.64 | 0.9967 | Excellent fit, captures curve |

**Polynomial regression is clearly superior** for this dataset!

---

## 📁 Project Structure

```
Polynomial_Regression_Project/
│
├── app.py                               # Main Streamlit web application
├── run_app.py                           # Auto-launcher script (opens Chrome)
├── run.bat                              # Windows batch file launcher
├── Temperature_vs_IceCreamSales.csv     # Dataset file (7 data points)
├── requirements.txt                     # Python package dependencies
└── README.md                            # This documentation file
```

### File Descriptions

- **`app.py`** - Main application with all functionality, UI, and ML models
- **`run_app.py`** - Launcher that automatically opens Chrome browser
- **`run.bat`** - Windows batch file for easy double-click launching
- **`Temperature_vs_IceCreamSales.csv`** - Dataset with temperature and sales data
- **`requirements.txt`** - List of required Python packages
- **`README.md`** - Complete project documentation

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|:-----------|:--------|:--------|
| **Python** | 3.7+ | Programming language |
| **NumPy** | ≥1.21.0 | Numerical computations |
| **Pandas** | ≥1.3.0 | Data manipulation and CSV handling |
| **Matplotlib** | ≥3.4.0 | Data visualization and plotting |
| **scikit-learn** | ≥0.24.0 | Machine learning models (Linear & Polynomial Regression) |
| **Streamlit** | ≥1.28.0 | Web application framework |

---

## 🎨 UI/UX Features

### Design Elements
- **Gradient Theme:** Purple-to-pink gradient color scheme
- **Glass Morphism:** Modern card designs with backdrop blur
- **Smooth Animations:** Slide-in and fade effects
- **Custom Fonts:** Poppins font family for professional look
- **Responsive Design:** Adapts to different screen sizes

### Interactive Elements
- **Temperature Slider:** Visual input with gradient track
- **Number Input:** Alternative text input with focus effects
- **Predict Button:** Large, prominent button with gradient background
- **Model Radio Buttons:** Clean selection interface
- **Hover Effects:** Cards lift and scale on hover

---

## 🔍 Troubleshooting

### Common Issues

**Issue:** Chrome doesn't open automatically
- **Solution:** Manually open Chrome and navigate to `http://localhost:8501`

**Issue:** Import errors when running
- **Solution:** Run `pip install -r requirements.txt` to install dependencies

**Issue:** CSV file not found
- **Solution:** Ensure `Temperature_vs_IceCreamSales.csv` is in the same directory as `app.py`

**Issue:** Port 8501 already in use
- **Solution:** Stop other Streamlit apps or use: `streamlit run app.py --server.port 8502`

**Issue:** App runs slowly
- **Solution:** The app uses caching for optimization. First load may be slower.

---

## 📝 Code Features

### Optimizations
- ✅ **Streamlit Caching:** `@st.cache_data` for model and plot caching
- ✅ **Efficient Plotting:** Base plots cached, only highlights updated
- ✅ **Session State:** Manages prediction state efficiently
- ✅ **Error Handling:** Comprehensive try-catch blocks

### Best Practices
- ✅ **Clean Code:** Well-organized classes and functions
- ✅ **Documentation:** Comprehensive docstrings
- ✅ **Type Safety:** Proper data type handling
- ✅ **User Feedback:** Clear error messages and status indicators

---

## 🎓 Educational Value

This project demonstrates:
- ✅ **Regression Analysis:** Linear and Polynomial regression implementation
- ✅ **Model Evaluation:** MSE and R² score calculation and interpretation
- ✅ **Data Visualization:** Professional plotting with Matplotlib
- ✅ **Web Development:** Streamlit web app creation
- ✅ **UI/UX Design:** Modern, responsive interface design
- ✅ **Machine Learning:** End-to-end ML pipeline

---

## 📊 Expected Results

When you run the application, you should see:

1. **Model Performance:**
   - Linear Regression: MSE ≈ 8,588, R² ≈ 0.9485
   - Polynomial Regression: MSE ≈ 545, R² ≈ 0.9967

2. **Sample Predictions:**
   - 25°C → ~₹346 (Polynomial) / ~₹426 (Linear)
   - 30°C → ~₹747 (Polynomial) / ~₹823 (Linear)
   - 35°C → ~₹1,355 (Polynomial) / ~₹1,220 (Linear)

3. **Visualizations:**
   - Regression plot showing both models
   - Comparison charts for metrics
   - Highlighted prediction points

---

## 🔐 Security & Privacy

- ✅ **Local Processing:** All data processed locally, no external servers
- ✅ **No Data Collection:** No user data is collected or stored
- ✅ **Open Source:** Code is transparent and auditable

---

## 📄 License

This project is for **educational purposes only**.

---

## 👨‍💻 Author

**B.TECH - IT - 3 Year - A Section**  
**Subject:** Machine Learning (22IT307)  
**Module:** Module2-T1

---

## 🙏 Acknowledgments

- **scikit-learn** for machine learning algorithms
- **Streamlit** for the web framework
- **Matplotlib** for visualization capabilities
- **Pandas & NumPy** for data processing

---

## 📞 Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review the **sidebar** in the application for help
3. Ensure all dependencies are installed correctly

---

## 🚀 Quick Start

### For New Users (Clone from GitHub)

```bash
# 1. Clone the repository
git clone https://github.com/Venkatatejadegala/Sales-Prediction.git

# 2. Navigate to project directory
cd Sales-Prediction

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python run_app.py

# 5. Use the app in Chrome (opens automatically)
```

### For Existing Users

```bash
# 1. Install dependencies (if not already installed)
pip install -r requirements.txt

# 2. Run the application
python run_app.py

# 3. Use the app in Chrome (opens automatically)
```

**That's it!** The application will open in your browser and you can start predicting ice cream sales! 🍦

### Repository Link
🔗 **GitHub:** [https://github.com/Venkatatejadegala/Sales-Prediction](https://github.com/Venkatatejadegala/Sales-Prediction)

---

**Made with ❤️ for Machine Learning Education**
