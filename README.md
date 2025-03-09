# Time Series Forecasting & Portfolio Optimization

## About the Project
This project provides a comprehensive framework for **Time Series Forecasting** and **Portfolio Optimization** using Python. It is designed for financial analysts, data scientists, and researchers interested in forecasting stock prices, evaluating market risks, and optimizing asset allocations using various statistical and machine learning techniques.

### **Key Features:**
- **Time Series Forecasting:**
  - Stationarity checks (ADF test)
  - Time series decomposition
  - ARIMA model training and evaluation
  - Future predictions with visualization
- **Portfolio Optimization:**
  - Mean-Variance Optimization (Markowitz Model)
  - Efficient Frontier Calculation
  - Risk-return tradeoff analysis
  - Asset allocation optimization

## Built With
The project is built using the following libraries and frameworks:
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Statsmodels
- Scikit-learn
- SciPy
- PyPortfolioOpt

## Prerequisites
Before using this project, ensure you have:
- Python 3.8+
- Jupyter Notebook or any Python IDE
- Virtual environment (optional but recommended)

## Installation
### **Step 1: Clone the Repository**
```sh
git clone https://github.com/your-repo/time-series-portfolio-optimization.git
cd time-series-portfolio-optimization
```
### **Step 2: Create a Virtual Environment (Optional but Recommended)**
```sh
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
```
### **Step 3: Install Dependencies**
```sh
pip install -r requirements.txt
```

## Usage
### **1. Time Series Forecasting**
Load the time series forecasting module in Jupyter Notebook:
```python
from time_series_forecaster import TimeSeriesForecaster
import pandas as pd

# Load data
prices = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)
returns = prices.pct_change().dropna()

# Initialize forecaster
forecaster = TimeSeriesForecaster("TSLA", prices, returns)

# Check stationarity
forecaster.check_stationarity(forecaster.prices)
forecaster.check_stationarity(forecaster.returns)

# Fit ARIMA model
forecaster.fit_arima(p=2, d=1, q=1)

# Forecast and evaluate
forecaster.forecast_and_evaluate()
```
### **2. Portfolio Optimization**
Run the portfolio optimization script:
```python
from portfolio_optimizer import PortfolioOptimizer

# Load historical stock data
asset = ["SPY", "TSLA", "BND"]
data = pd.read_csv("asset.csv", index_col="Date", parse_dates=True)

# Initialize optimizer
optimizer = PortfolioOptimizer(stocks, data)

# Optimize portfolio
optimized_weights = optimizer.optimize()
print("Optimized Portfolio Weights:", optimized_weights)

# Visualize Efficient Frontier
optimizer.plot_efficient_frontier()
```

## Contact
For any questions or collaborations, reach out via:
- Email: zeynebmu40@gmail.com
- LinkedIn: (https://www.linkedin.com/in/zeyneba-mulat/)
- GitHub: (https://github.com/Zeyni12/Portifolio_Optimaization)

## Acknowledgements
Special thanks to:
- The **Statsmodels** and **Scikit-learn** teams for statistical modeling and ML tools
- The **PyPortfolioOpt** library for portfolio optimization
- Open-source contributors whose work inspired this project


