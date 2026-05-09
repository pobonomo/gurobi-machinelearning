import os
import xgboost as xgb
import numpy as np
from sklearn import datasets
from ..fixed_formulation import FixedRegressionModel

class TestXGBMisic(FixedRegressionModel):
    """Test the Mišić formulation for XGBoost."""

    def test_diabetes_xgboost_misic(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        xgb_reg = xgb.XGBRegressor(n_estimators=5, max_depth=3)
        xgb_reg.fit(X, y)
        one_case = {"predictor": xgb_reg, "nonconvex": 0}

        # Test with misic formulation
        self.do_one_case(one_case, X, 3, formulation="misic", float_type=np.float32)

    def test_diabetes_xgboost_misic_safety_floor(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        xgb_reg = xgb.XGBRegressor(n_estimators=5, max_depth=3)
        xgb_reg.fit(X, y)
        one_case = {"predictor": xgb_reg, "nonconvex": 0}

        # Test with misic formulation and safety_floor
        self.do_one_case(one_case, X, 3, formulation="misic", safety_floor=1e-5, float_type=np.float32)
