import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Avocado Setup ---
def setup_avocado():
    data_url = "https://raw.githubusercontent.com/Gurobi/modeling-examples/master/price_optimization/"
    avocado = pd.read_csv(data_url + "HABdata_2019_2022.csv")
    avocado_old = pd.read_csv(data_url + "kaggledata_till2018.csv")
    avocado["date"] = pd.to_datetime(avocado["date"], format="%m/%d/%y %H:%M")
    avocado_old["date"] = pd.to_datetime(avocado_old["date"], format="%m/%d/%y")
    avocado = pd.concat([avocado, avocado_old])
    avocado["year"] = pd.DatetimeIndex(avocado["date"]).year
    avocado["month"] = pd.DatetimeIndex(avocado["date"]).month
    peak_months = range(2, 8)
    avocado["peak"] = avocado["month"].apply(lambda x: 1 if x in peak_months else 0)
    avocado["units_sold"] = avocado["units_sold"] / 1000000
    avocado = avocado[avocado["type"] == "Conventional"]
    regions = ["Great_Lakes", "Midsouth", "Northeast", "West"]
    df = avocado[avocado.region.isin(regions)]
    
    X = df[["region", "price", "year", "peak"]]
    y = df["units_sold"]
    feat_transform = make_column_transformer(
        (OneHotEncoder(drop="first"), ["region"]),
        (StandardScaler(), ["price", "year"]),
        ("passthrough", ["peak"]),
        verbose_feature_names_out=False,
        remainder="drop",
    )
    rf_reg = make_pipeline(feat_transform, RandomForestRegressor(n_estimators=100, max_depth=8, random_state=1))
    rf_reg.fit(X, y)
    
    return rf_reg, regions

# --- Janos Setup ---
def get_janos_data():
    janos_data_url = "https://raw.githubusercontent.com/INFORMSJoC/2020.1023/master/data/"
    historical_data = pd.read_csv(janos_data_url + "college_student_enroll-s1-1.csv", index_col=0)
    studentsdata_full = pd.read_csv(janos_data_url + "college_applications6000.csv", index_col=0)
    return historical_data, studentsdata_full

def setup_janos_model(historical_data, features, n_estimators, max_depth):
    target = "enroll"
    regression = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=1)
    regression.fit(X=historical_data.loc[:, features], y=historical_data.loc[:, target])
    return regression

# --- MISIC Setup ---
def setup_misic_model(n_trees=100, max_depth=6, n_samples=2000, n_features=20):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples)

    model = lgb.train(
        {'objective': 'regression', 'num_leaves': 2**max_depth, 'max_depth': max_depth,
         'learning_rate': 0.1, 'verbose': -1, 'seed': 42},
        lgb.Dataset(X, y), num_boost_round=n_trees,
    )
    return model, n_features

# --- Issue 496 Setup ---
def setup_issue_496_model():
    np.random.seed(42)
    N = 1000
    X = np.random.randn(N, 5).astype(np.float32)
    X[:, 0] = 0.0
    active = np.random.rand(N) < 0.05
    X[active, 0] = np.random.choice([-5, 5], size=active.sum()).astype(np.float32)
    y = (X[:, 0] != 0).astype(np.int32)

    model = lgb.train(
        {'objective': 'binary', 'num_leaves': 4, 'max_depth': 2,
         'learning_rate': 0.3, 'verbose': -1, 'seed': 42,
         'deterministic': True, 'min_data_in_leaf': 10},
        lgb.Dataset(X, y), num_boost_round=1,
    )
    return model, 5
