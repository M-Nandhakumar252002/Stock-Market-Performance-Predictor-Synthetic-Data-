# Stock-Market-Performance-Predictor-Synthetic-Data-
## Overview
This project simulates stock price trends and election performance data to build a **predictive model** using Python.  
It uses synthetic time-series inspired data to mimic volatility and political impacts.

## Objectives
- Generate synthetic stock & election data.
- Apply regression and classification models:
  - Linear Regression
  - Random Forest
- Evaluate performance using RMSE, accuracy, and confusion matrix.

## Tools & Libraries
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## Example Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Synthetic Data
np.random.seed(42)
days = 365
data = pd.DataFrame({
    "day": np.arange(days),
    "stock_price": np.random.normal(100, 10, days).cumsum(),
    "election_factor": np.random.choice([0,1], size=days)
})

# Train/Test Split
X = data[["day", "election_factor"]]
y = data["stock_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print("RMSE:", rmse)
