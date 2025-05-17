import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load dataset
df = pd.read_csv("flower_dataset_20000.csv")

# Use last column as target if 'target' not found
if 'target' in df.columns:
    X = df.drop("target", axis=1)
    y = df["target"]
else:
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

# Split and scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Feature importance (absolute value of coefficients)
importances = np.abs(model.coef_)
features = np.array(X.columns)
indices = np.argsort(importances)[::-1]

# GUI with Next button to show 5 plots sequentially
figures = []

# 1. Feature Importance
fig1, ax1 = plt.subplots()
sns.barplot(x=importances[indices], y=features[indices], ax=ax1)
ax1.set_title("Feature Importance (Coefficients)")
figures.append(fig1)

# 2. Confusion Matrix - Not applicable for regression, so we use residual distribution
fig2, ax2 = plt.subplots()
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, ax=ax2, color='purple')
ax2.set_title("Residuals Distribution")
figures.append(fig2)

# 3. ROC Curve - Not applicable for regression, plot predicted vs actual
fig3, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred, alpha=0.3)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax3.set_title("Actual vs Predicted")
ax3.set_xlabel("Actual")
ax3.set_ylabel("Predicted")
figures.append(fig3)

# 4. Prediction Histogram
fig4, ax4 = plt.subplots()
ax4.hist(y_pred, bins=20, color='skyblue', edgecolor='black')
ax4.set_title("Prediction Probability Histogram")
figures.append(fig4)

# 5. Line Plot of predictions (sample)
fig5, ax5 = plt.subplots()
sample_idx = np.arange(100)
ax5.plot(sample_idx, y_test[:100], label='Actual', marker='o')
ax5.plot(sample_idx, y_pred[:100], label='Predicted', marker='x')
ax5.set_title("Actual vs Predicted (First 100 samples)")
ax5.legend()
figures.append(fig5)

# TK window for plots
current = 0
root = tk.Tk()
root.title("Linear Regression Visualizations")

canvas = FigureCanvasTkAgg(figures[current], master=root)
canvas.get_tk_widget().pack()
canvas.draw()

def next_plot():
    global current
    current += 1
    if current < len(figures):
        canvas.figure = figures[current]
        canvas.draw()
    else:
        next_button.config(state='disabled')

next_button = tk.Button(root, text="Next", command=next_plot)
next_button.pack()

root.mainloop()
