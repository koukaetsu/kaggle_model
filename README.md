# 🏠 House Price Prediction Competition (RMSE: 0.0095)

This repository contains my full pipeline for solving a house price prediction challenge. The task was to model structured housing data and predict final sale prices using regression methods.

I approached the problem in two stages:
1. A custom two-phase Random Forest tuning strategy using gradient and spline-based optimization.
2. A final XGBoost model fine-tuned using Bayesian hyperparameter search.

## 🧠 Project Highlights

- 🎯 Task: Predict SalePrice based on housing features
- 📈 Metric: Root Mean Squared Error (RMSE)
- 🧮 Final Result: **RMSE = 0.0095**

---

## 📂 Key Files

- `competition_rf_own.py`:  
  Implements a **custom tuning algorithm** for Random Forests using:
  - Stage 1: `Uspline()` — uses Univariate Spline to approximate the MAE curve and locate local minima of tree_leaf parameters.
  - Stage 2: `gradients()` — applies `np.gradient` and `np.sign` changes to find best turning points of MAE values.
  - These methods are used to **intelligently search optimal number of tree leaf nodes**, rather than brute-force grid search.

- `competition_xboost_bys.py`:  
  Final model using **XGBoost + Bayesian Optimization** (via `BayesSearchCV` from `skopt`), with:
  - Categorical feature encoding via `TargetEncoder`
  - Log-transformation of SalePrice for normalization
  - No spline or derivative method applied here — only probabilistic search

---

## 🛠️ Techniques Used

| Category | Methods |
|---------|---------|
| **Modeling** | RandomForest, XGBoost |
| **Custom Tuning** | Spline interpolation, Gradient-based MAE minimization |
| **Hyperparameter Optimization** | Bayesian search (skopt) |
| **Feature Engineering** | Target Encoding, Feature Subset |
| **Evaluation** | Cross-Validation, MAE & RMSE |

