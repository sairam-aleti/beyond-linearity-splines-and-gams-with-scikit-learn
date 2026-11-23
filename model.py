"""
Beyond Linearity: Splines and GAMs with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_wage
import os
import tempfile
import urllib.request
import pandas as pd

WAGE_URL = "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"

def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        'n': len(df),
        'columns': list(df.columns),
        'age_range': (int(df['age'].min()), int(df['age'].max())),
        'wage_mean': round(df['wage'].mean(), 2),
        'n_education_levels': df['education'].nunique()
    }

# Step 2 - split_wage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    return train, test


def age_xy(df):
    X = df[['age']]
    y = df['wage']
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        'age': np.linspace(lo, hi, n)
    })

# Step 3 - cv_tools
import numpy as np
from sklearn.model_selection import cross_val_score

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring='neg_mean_squared_error'
    )

    # Convert negative MSE scores to positive MSE
    fold_mses = -scores

    mean = np.mean(fold_mses)
    se = np.std(fold_mses, ddof=1) / np.sqrt(len(fold_mses))

    return round(mean, 1), round(se, 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean, se = cv_mse(model, X, y, cv)

        means.append(mean)
        ses.append(se)

    return means, ses


def one_se_rule(values, means, ses, prefer='smaller'):
    # Find the best (lowest) cross-validation MSE
    best_idx = np.argmin(means)
    threshold = means[best_idx] + ses[best_idx]

    # Select values within one standard error of the best
    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == 'smaller':
        return min(eligible)
    elif prefer == 'larger':
        return max(eligible)
    else:
        raise ValueError("prefer must be 'smaller' or 'larger'")

# Step 4 - polynomial_regression
import numpy as np
from scipy.stats import f as f_dist
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def poly_model(degree):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression()
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for d in range(2, max_degree + 1):
        model_prev = poly_model(d - 1)
        model_curr = poly_model(d)

        model_prev.fit(X, y)
        model_curr.fit(X, y)

        rss_prev = np.sum((y - model_prev.predict(X)) ** 2)
        rss_curr = np.sum((y - model_curr.predict(X)) ** 2)

        F = ((rss_prev - rss_curr) / 1) / (rss_curr / (n - d - 1))
        p = f_dist.sf(F, 1, n - d - 1)

        results.append((d, round(F, 2), round(p, 4)))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, ses = poly_curve(X, y, degrees, cv)

    degree_min = degrees[np.argmin(means)]

    anova_results = anova_degrees(X, y, max(degrees))

    degree_anova = 1
    for d, F, p in anova_results:
        if p < alpha:
            degree_anova = d
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    return np.round(model.predict(grid).ravel(), 2)

# Step 5 - step_functions
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode='onehot-dense',
            strategy='uniform'
        ),
        LinearRegression()
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)

    bins_min = bins[np.argmin(means)]
    bins_1se = one_se_rule(bins, means, ses, prefer='smaller')

    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps['kbinsdiscretizer']
    return np.round(discretizer.bin_edges_[0], 1).tolist()


def step_levels(model, grid):
    predictions = model.predict(grid)
    return sorted(np.round(np.unique(predictions), 1).tolist())

# Step 6 - spline_regression (not yet solved)
# TODO: implement

# Step 7 - extrapolation (not yet solved)
# TODO: implement

# Step 8 - smoothing_spline (not yet solved)
# TODO: implement

# Step 9 - local_smoother (not yet solved)
# TODO: implement

# Step 10 - gam_pipeline (not yet solved)
# TODO: implement

# Step 11 - partial_effects (not yet solved)
# TODO: implement

# Step 12 - logistic_gam (not yet solved)
# TODO: implement

# Step 13 - fit_age_models (not yet solved)
# TODO: implement

# Step 14 - test_comparison (not yet solved)
# TODO: implement

