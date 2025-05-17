import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Generate synthetic classification data (3 classes, 20 features)
X, y = make_classification(
    n_samples=20000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

# Convert to DataFrame and add target column 'species'
df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])
df['species'] = y

# Features and target
X = df.drop('species', axis=1)
y = df['species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train SVM Classifier with probability estimates enabled
model = SVC(probability=True, random_state=42)
model.fit(X_train, y_train)

# Predictions and probabilities
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# No built-in feature importance, so placeholder zeros
importances = np.zeros(X.shape[1])
features = np.array(X.columns)

# Prepare figures
figures = []

# 1. Feature Importance Placeholder Bar Plot
fig1, ax1 = plt.subplots()
sns.barplot(x=importances, y=features, ax=ax1, palette="viridis")
ax1.set_title("Feature Importance (SVM has no coefficients in this form)")
figures.append(fig1)

# 2. Confusion Matrix Heatmap
fig2, ax2 = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title("Confusion Matrix")
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
figures.append(fig2)

# 3. ROC Curve (One-vs-Rest)
fig3, ax3 = plt.subplots()
for i in range(y_proba.shape[1]):
    fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
ax3.plot([0, 1], [0, 1], 'k--')
ax3.set_title("ROC Curve (One-vs-Rest)")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend()
figures.append(fig3)

# 4. Prediction Probability Histogram for Class 0
fig4, ax4 = plt.subplots()
ax4.hist(y_proba[:, 0], bins=20, color='skyblue', edgecolor='black')
ax4.set_title("Prediction Probability Histogram (Class 0)")
ax4.set_xlabel("Predicted Probability")
ax4.set_ylabel("Frequency")
figures.append(fig4)

# 5. Actual vs Predicted Labels Line Plot (First 100 samples)
fig5, ax5 = plt.subplots()
ax5.plot(range(100), y_test.iloc[:100], label='Actual', marker='o')
ax5.plot(range(100), y_pred[:100], label='Predicted', marker='x')
ax5.set_title("Actual vs Predicted Labels (First 100 samples)")
ax5.set_xlabel("Sample Index")
ax5.set_ylabel("Class Label")
ax5.legend()
figures.append(fig5)

# Tkinter GUI for displaying plots one by one
current = 0
root = tk.Tk()
root.title("SVM Classifier Visualizations")

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
