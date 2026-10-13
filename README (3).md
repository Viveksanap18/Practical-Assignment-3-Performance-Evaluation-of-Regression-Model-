# Practical Assignment 3: Performance Evaluation of Regression Model

**TY B.Tech. Artificial Intelligence and Machine Learning**

| | |
|---|---|
| **Student** | Vivek Vilas Sanap |
| **PRN** | 202401040091 |
| **Roll No.** | 091 |
| **Branch** | Computer Engineering |
| **Division / Batch** | B / A1 |
| **Faculty** | Dr. Khushal Khairnar |
| **Date of Submission** | 01/10/2026 |

## Overview

This practical builds regression models to predict the **price of a car** from its specifications, evaluates them with standard regression metrics, and implements **Linear Regression using Gradient Descent from scratch**, comparing it with Scikit-learn.

**Objectives**

1. Build regression models that predict a continuous value (car price in US dollars).
2. Evaluate the models using MAE, MSE, RMSE and R² score.
3. Study how the learning rate and number of iterations affect Gradient Descent convergence.
4. Compare the Gradient Descent model with Scikit-learn Linear Regression.

## Dataset

`CarPrice.csv` is the Car Price Assignment dataset.

- 205 cars, 26 columns, no missing values
- Target: `price` (US dollars, range $5,118 to $45,400)
- Inputs: 14 numeric columns (engine size, curb weight, horsepower, mileage and others) and 9 categorical columns (fuel type, body style, drive wheel, number of cylinders and others)
- `car_ID` and `CarName` are dropped; categorical columns are one-hot encoded, giving 43 input features
- Split: 80:20 with `random_state=42` (164 training cars, 41 test cars)

## Models

| Model | Notes |
|---|---|
| Simple Linear Regression | Engine size only |
| Multiple Linear Regression | All 43 features |
| Ridge Regression | Standardized features; alpha chosen by 5-fold cross-validation on the training data (alpha = 10) |
| Polynomial Regression | Engine size only; degree chosen by cross-validation on the training data (degree 3) |
| Gradient Descent (from scratch) | Standardized engine size, learning rate 0.1, 500 iterations |

## Results (test set, 41 cars)

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Simple Linear Regression | 2,748.69 | 3,932.61 | 0.8041 |
| Multiple Linear Regression | 2,089.38 | 2,912.39 | 0.8926 |
| Ridge Regression (alpha = 10) | 2,271.35 | 3,377.92 | 0.8555 |
| Polynomial Regression (degree 3) | 2,348.84 | 3,409.02 | 0.8528 |
| Gradient Descent (engine size) | 2,748.69 | 3,932.61 | 0.8041 |
| Scikit-learn LR (engine size) | 2,748.69 | 3,932.61 | 0.8041 |

**5-fold cross-validation R² (shuffled folds):** Multiple Linear Regression averages 0.8633 and Ridge averages 0.8822. The test set is small, so cross-validation is the more reliable comparison.

**Gradient Descent findings**

- Final parameters (standardized engine size): slope = 6691.0726, intercept = 13223.4146. They match Scikit-learn exactly.
- The cost came within 1% of its final value after 17 iterations at learning rate 0.1.
- Learning rate 0.001 is too slow, 0.01 is still converging after 100 iterations, 0.1 and 0.5 converge quickly, and 1.05 diverges.

The full report, with theory, test cases, graphs and analysis, is in the submitted assignment document.

## Project structure

```
.
├── car_price_regression.py   # Complete implementation (models, metrics, Gradient Descent, graphs)
├── CarPrice.csv              # Dataset
└── README.md
```

Running the script also creates `results.json` (all metrics) and six graphs:

- `graph1_price_vs_enginesize.png`
- `graph2_actual_vs_predicted.png`
- `graph3_residuals.png`
- `graph4_gd_cost.png`
- `graph5_learning_rates.png`
- `graph6_model_comparison.png`

## How to run

Requires Python 3 with NumPy, Pandas, Matplotlib and Scikit-learn (tested with scikit-learn 1.9.1 and pandas 3.0.5).

```bash
pip install numpy pandas matplotlib scikit-learn
python car_price_regression.py
```

Keep `CarPrice.csv` in the same folder as the script. The metrics are printed to the terminal and the graphs are saved as PNG files in the same folder.

## Tools

Python 3, VS Code / Jupyter Notebook, NumPy, Pandas, Matplotlib, Scikit-learn.

## Declaration

This practical work was completed by me, and the implementation, analysis and answers are my own work.
