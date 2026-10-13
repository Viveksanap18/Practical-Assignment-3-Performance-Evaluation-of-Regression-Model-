import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})
BLUE, ORANGE, GREEN, RED = "#2a6fb0", "#e07b24", "#3a9a5b", "#c0392b"

results = {}

# ---------------------------------------------------------------- 1. Data
df = pd.read_csv("CarPrice.csv")
results["shape"] = list(df.shape)
print("Dataset shape:", df.shape)
print(df[["CarName", "enginesize", "horsepower", "curbweight", "price"]].head())

df = df.drop(columns=["car_ID", "CarName"])
X = pd.get_dummies(df.drop(columns=["price"]), drop_first=True).astype(float)
y = df["price"]
results["n_features"] = X.shape[1]
results["n_numeric"] = int(df.drop(columns=["price"]).select_dtypes("number").shape[1])
print("Features after one-hot encoding:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
results["n_train"], results["n_test"] = len(X_train), len(X_test)
print("Train / test:", len(X_train), len(X_test))


def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(actual, predicted)
    print(f"{name:34s} MAE={mae:10.2f} MSE={mse:14.2f} RMSE={rmse:9.2f} R2={r2:.4f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------- 2. Models
# (a) Simple linear regression: engine size only
simple = LinearRegression().fit(X_train[["enginesize"]], y_train)
simple_pred = simple.predict(X_test[["enginesize"]])
results["simple"] = evaluate("Simple LR (enginesize)", y_test, simple_pred)

# (b) Multiple linear regression: all features
multi = LinearRegression().fit(X_train, y_train)
multi_pred = multi.predict(X_test)
results["multiple"] = evaluate("Multiple Linear Regression", y_test, multi_pred)
# The rows of this dataset are ordered by car make, so folds must be shuffled
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
results["multiple_cv"] = {"folds": cv.tolist(), "mean": float(cv.mean()), "std": float(cv.std())}
print("Multiple LR 5-fold CV R2:", cv.round(4), "mean", round(cv.mean(), 4))

# (c) Ridge regression (standardised features); alpha chosen by CV on training data only
alpha_cv = {}
for a in (0.1, 1, 10, 30, 100):
    s = cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=a)),
                        X_train, y_train, cv=kf, scoring="r2")
    alpha_cv[a] = float(s.mean())
    print("ridge alpha", a, "CV R2", round(s.mean(), 4))
best_alpha = max(alpha_cv, key=alpha_cv.get)
results["alpha_cv"] = {str(k): v for k, v in alpha_cv.items()}
results["best_alpha"] = best_alpha
ridge = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)).fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
results["ridge"] = evaluate(f"Ridge Regression (alpha={best_alpha})", y_test, ridge_pred)
rcv = cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)), X, y, cv=kf, scoring="r2")
results["ridge_cv"] = {"folds": rcv.tolist(), "mean": float(rcv.mean()), "std": float(rcv.std())}
print("Ridge 5-fold CV R2:", rcv.round(4), "mean", round(rcv.mean(), 4))

# (d) Polynomial regression on engine size; degree chosen by CV on training data only
poly_cols = ["enginesize"]
degree_scores = {}
for d in (1, 2, 3, 4, 5):
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())
    cvs = cross_val_score(m, X_train[poly_cols], y_train, cv=kf, scoring="r2").mean()
    m.fit(X_train[poly_cols], y_train)
    degree_scores[d] = {
        "cv_R2": float(cvs),
        "train_R2": r2_score(y_train, m.predict(X_train[poly_cols])),
        "test_R2": r2_score(y_test, m.predict(X_test[poly_cols]))}
    print("poly degree", d, {k: round(v, 4) for k, v in degree_scores[d].items()})
results["poly_degrees"] = degree_scores
best_deg = max(degree_scores, key=lambda d: degree_scores[d]["cv_R2"])
results["best_degree"] = best_deg
poly = make_pipeline(PolynomialFeatures(best_deg), StandardScaler(), LinearRegression())
poly.fit(X_train[poly_cols], y_train)
poly_pred = poly.predict(X_test[poly_cols])
results["poly2"] = evaluate(f"Polynomial Regression (deg {best_deg})", y_test, poly_pred)

# ---------------------------------------------------------------- 3. Gradient descent from scratch
feat = "enginesize"
x_tr, x_te = X_train[feat].values, X_test[feat].values
mu, sd = x_tr.mean(), x_tr.std()
xs_tr, xs_te = (x_tr - mu) / sd, (x_te - mu) / sd
yt = y_train.values
n = len(xs_tr)


def gradient_descent(lr, iters):
    m, c = 0.0, 0.0
    history = []
    for _ in range(iters):
        err = (m * xs_tr + c) - yt
        m -= lr * (2 / n) * np.sum(err * xs_tr)
        c -= lr * (2 / n) * np.sum(err)
        history.append(float(np.mean((yt - (m * xs_tr + c)) ** 2)))
    return m, c, history


LR, ITERS = 0.1, 500
slope, intercept, cost_hist = gradient_descent(LR, ITERS)
gd_pred = slope * xs_te + intercept
results["gd"] = evaluate("Gradient Descent (enginesize)", y_test, gd_pred)
results["gd_params"] = {"slope": slope, "intercept": intercept,
                        "final_train_mse": cost_hist[-1], "lr": LR, "iters": ITERS,
                        "initial_mse": cost_hist[0]}
print("GD slope, intercept, final MSE:", round(slope, 4), round(intercept, 4), round(cost_hist[-1], 2))

# first iteration where cost is within 1% of final value
conv_iter = next(i for i, c in enumerate(cost_hist, 1) if c <= cost_hist[-1] * 1.01)
results["gd_params"]["converged_at"] = conv_iter

sk = LinearRegression().fit(x_tr.reshape(-1, 1), yt)
sk_pred = sk.predict(x_te.reshape(-1, 1))
results["sk_single"] = evaluate("Scikit-learn LR (enginesize)", y_test, sk_pred)
# convert sklearn raw coefficients to standardised space for comparison
sk_slope_std = sk.coef_[0] * sd
sk_intercept_std = sk.coef_[0] * mu + sk.intercept_
results["sk_params"] = {"slope_std": float(sk_slope_std), "intercept_std": float(sk_intercept_std),
                        "slope_raw": float(sk.coef_[0]), "intercept_raw": float(sk.intercept_)}
print("sklearn (std space) slope, intercept:", round(sk_slope_std, 4), round(sk_intercept_std, 4))
results["gd_vs_sk_max_pred_diff"] = float(np.max(np.abs(gd_pred - sk_pred)))

# learning-rate experiment
lr_study = {}
for lr in (0.001, 0.01, 0.1, 0.5, 1.05):
    m, c, h = gradient_descent(lr, 100)
    lr_study[str(lr)] = {"cost_at_10": h[9], "cost_at_100": h[-1], "history": h,
                         "diverged": bool(h[-1] > h[0])}
    print("lr", lr, "cost@10", f"{h[9]:.4g}", "cost@100", f"{h[-1]:.4g}")
results["lr_study"] = {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                       for k, v in lr_study.items()}

# iterations experiment
iter_study = {}
for it in (10, 50, 100, 500, 2000):
    m, c, h = gradient_descent(LR, it)
    iter_study[str(it)] = {"train_mse": h[-1],
                           "test_r2": float(r2_score(y_test, m * xs_te + c))}
results["iter_study"] = iter_study
print("iteration study", iter_study)

# example prediction
example_size = 130
ex_scaled = (example_size - mu) / sd
ex_pred = slope * ex_scaled + intercept
results["example"] = {"enginesize": example_size, "scaled": float(ex_scaled), "price": float(ex_pred),
                      "mean_enginesize": float(mu), "std_enginesize": float(sd)}
mean_pred = slope * 0.0 + intercept
results["example"]["price_at_mean"] = float(mean_pred)
print("Example: engine 130 ->", round(ex_pred, 2), "| scaled", round(ex_scaled, 4))

# feature importance for report (top coefficients, standardised)
std_multi = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
coefs = pd.Series(std_multi[-1].coef_, index=X.columns)
top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(6)
results["top_coefs"] = {k: float(v) for k, v in top.items()}
print(top.round(1))

corr = df.select_dtypes("number").corr()["price"].drop("price")
results["top_corr"] = {k: float(v) for k, v in corr.reindex(corr.abs().sort_values(ascending=False).index).head(5).items()}
results["price_stats"] = {"mean": float(y.mean()), "min": float(y.min()), "max": float(y.max()),
                          "median": float(y.median())}

# ---------------------------------------------------------------- 4. Graphs
# Graph 1 - price vs engine size with GD fit
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.scatter(x_tr, yt, s=22, color=BLUE, alpha=0.6, label="Training cars")
ax.scatter(x_te, y_test, s=26, color=ORANGE, label="Test cars")
grid = np.linspace(x_tr.min(), x_tr.max(), 100)
ax.plot(grid, slope * ((grid - mu) / sd) + intercept, color=RED, lw=2,
        label="Gradient Descent fit")
ax.set_xlabel("Engine size (cu in)")
ax.set_ylabel("Price (USD)")
ax.set_title("Car Price vs Engine Size")
ax.legend()
fig.tight_layout(); fig.savefig("graph1_price_vs_enginesize.png"); plt.close(fig)

# Graph 2 - actual vs predicted (multiple linear regression)
fig, ax = plt.subplots(figsize=(6, 5.2))
ax.scatter(y_test, multi_pred, color=BLUE, s=30, label="Predicted values")
lo, hi = min(y_test.min(), multi_pred.min()), max(y_test.max(), multi_pred.max())
ax.plot([lo, hi], [lo, hi], "--", color=RED, label="Perfect prediction")
ax.set_xlabel("Actual price (USD)")
ax.set_ylabel("Predicted price (USD)")
ax.set_title("Actual vs Predicted Car Prices (Multiple Linear Regression)")
ax.legend()
fig.tight_layout(); fig.savefig("graph2_actual_vs_predicted.png"); plt.close(fig)

# Graph 3 - residuals
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.scatter(multi_pred, y_test.values - multi_pred, color=BLUE, s=30)
ax.axhline(0, color=RED, ls="--")
ax.set_xlabel("Predicted price (USD)")
ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Residual Plot (Multiple Linear Regression)")
fig.tight_layout(); fig.savefig("graph3_residuals.png"); plt.close(fig)

# Graph 4 - GD cost curve
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(range(1, ITERS + 1), cost_hist, color=BLUE, lw=2)
ax.axvline(conv_iter, color=ORANGE, ls="--", label=f"Within 1% of final cost (iteration {conv_iter})")
ax.set_xlabel("Iterations")
ax.set_ylabel("Mean Squared Error")
ax.set_title(f"Gradient Descent Cost Reduction (learning rate = {LR})")
ax.legend()
fig.tight_layout(); fig.savefig("graph4_gd_cost.png"); plt.close(fig)

# Graph 5 - learning rate comparison
fig, ax = plt.subplots(figsize=(7, 4.6))
colors = [GREEN, BLUE, ORANGE, "#7d5ba6", RED]
for (lr, v), col in zip(lr_study.items(), colors):
    h = np.array(v["history"])
    ax.plot(range(1, 101), np.minimum(h, 1e9), color=col, lw=2, label=f"lr = {lr}")
ax.set_yscale("log")
ax.set_xlabel("Iterations")
ax.set_ylabel("Training MSE (log scale)")
ax.set_title("Effect of Learning Rate on Gradient Descent")
ax.legend(loc="center right")
fig.tight_layout(); fig.savefig("graph5_learning_rates.png"); plt.close(fig)

# Graph 6 - model comparison (R2 and RMSE)
names = ["Simple LR", "Multiple LR", "Ridge", f"Poly (deg {best_deg})", "GD (scratch)"]
r2s = [results[k]["R2"] for k in ("simple", "multiple", "ridge", "poly2", "gd")]
rmses = [results[k]["RMSE"] for k in ("simple", "multiple", "ridge", "poly2", "gd")]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
axes[0].bar(names, r2s, color=BLUE)
axes[0].set_title("R² Score by Model (higher is better)")
axes[0].set_ylim(0, 1)
for i, v in enumerate(r2s):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
axes[1].bar(names, rmses, color=ORANGE)
axes[1].set_title("RMSE by Model in USD (lower is better)")
for i, v in enumerate(rmses):
    axes[1].text(i, v + 60, f"{v:,.0f}", ha="center", fontsize=9)
for a in axes:
    a.tick_params(axis="x", rotation=25)
    a.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("graph6_model_comparison.png"); plt.close(fig)

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved results.json and 6 graphs")
