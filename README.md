# kaggle_model
# 🏠 House Price Prediction Competition (RMSE: 0.0095)

This repository contains my end-to-end solution to a regression-based house price prediction competition. I approached the task using both a self-designed Random Forest baseline and a final XGBoost model optimized with Bayesian search.

## 🧠 Project Highlights

- 📊 Dataset: Structured real estate data (train/test split provided)
- 🎯 Task: Predict final house prices (SalePrice) as a regression target
- 🧪 Evaluation Metric: Root Mean Squared Error (RMSE)
- 🧮 Final Model RMSE: **0.0095**

---

## 📂 Project Structure

- `competition_rf_own.py`:  
  My initial model design using RandomForestRegressor and simple preprocessing strategies. This baseline helped validate important features and spot basic patterns in the data.

- `competition_xboost_bys.py`:  
  My final optimized model using **XGBoost** and **Bayesian hyperparameter tuning** (`BayesSearchCV`). Includes:
  - Log-transform of `SalePrice` target for stabilization
  - Feature encoding via `TargetEncoder` for categoricals
  - Use of `scikit-optimize` for efficient hyperparameter search
  - Spline-based interpolation on missing data

---

## 🛠️ Techniques Used

| Category | Methods |
|---------|---------|
| **Models** | Random Forest, XGBoost |
| **Feature Handling** | Target Encoding, Feature Subset Selection |
| **Optimization** | Bayesian Search (skopt) |
| **Evaluation** | RMSE, Cross-validation |
| **Data Tricks** | Spline interpolation, Log-transformation |

---

## 📈 Results

| Model | RMSE | Notes |
|-------|------|-------|
| `RandomForestRegressor` (manual) | ~0.03 | First iteration, good baseline |
| `XGBoost + BayesSearchCV` | **0.0095** | Final model with full tuning and transformation pipeline |

---

## ✨ Reflections

This competition taught me not just about regression modeling, but also how powerful optimization and preprocessing can be. Starting with a basic Random Forest gave me a good intuition for feature importance and tree models, while XGBoost + Bayesian optimization helped me push the performance significantly further.

---

## 📌 Dependencies

- Python 3.8+
- `pandas`, `numpy`, `sklearn`, `xgboost`, `category_encoders`, `scikit-optimize`
