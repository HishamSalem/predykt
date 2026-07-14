"""Shared fixtures for the predykt test suite."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def binary_data():
    """Numeric + categorical features, binary target with a true num1*num2
    interaction and a categorical main effect. NaNs present in cat1."""
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({
        "num1": rng.normal(size=n),
        "num2": rng.normal(size=n),
        "cat1": rng.choice(["a", "b", "c", None], size=n, p=[.4, .3, .2, .1]),
        "cat2": pd.Categorical(rng.choice(["x", "y"], size=n)),
    })
    logit = (0.8 * df["num1"] + 0.6 * df["num1"] * df["num2"]
             + 0.7 * (df["cat1"] == "a").astype(float)
             + rng.normal(scale=1.0, size=n))
    y = (logit > 0.4).astype(int).to_numpy()
    reps = pd.DataFrame({"num1_x_num2": df["num1"] * df["num2"]},
                        index=df.index)
    return df, y, reps


@pytest.fixture(scope="session")
def numeric_data():
    """Numeric-only features with a true x0*x1 interaction in the target."""
    rng = np.random.default_rng(1)
    n = 1200
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["x0", "x1", "x2"])
    logit = X["x0"] + 0.9 * X["x0"] * X["x1"] + rng.normal(scale=1.0, size=n)
    y = (logit > 0).astype(int).to_numpy()
    return X, y
