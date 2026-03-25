# model.py

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# 1. LOAD DATA
# =========================
data = pd.read_csv("datanew.csv")

# =========================
# 2. CLEAN DATA
# =========================
data = data.dropna()

data = data.drop([
    "id",
    "date",
    "street",
    "city",
    "statezip",
    "country"
], axis=1, errors='ignore')

# =========================
# 3. FEATURE SELECTION
# =========================
features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated"
]

X = data[features]
y = data["price"]

# =========================
# 4. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. TRAIN MODEL
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# 6. PREDICT
# =========================
y_pred = model.predict(X_test)

# =========================
# 7. EVALUATE
# =========================
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# =========================
# 8. SAVE MODEL (.pkl)
# =========================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")

# ===========================
# 9. VISUALIZING MODEL
# ===========================
import matplotlib.pyplot as plt

# Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue')
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()