import os
import lightgbm as lgb
import numpy as np
from sklearn import datasets
from ..fixed_formulation import FixedRegressionModel

class TestLGBMMisic(FixedRegressionModel):
    """Test the Mišić formulation for LightGBM."""

    def test_diabetes_lightgbm_misic(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        lgbm_reg = lgb.sklearn.LGBMRegressor(n_estimators=5, max_depth=3)
        lgbm_reg.fit(X, y)
        one_case = {"predictor": lgbm_reg, "nonconvex": 0}

        # Test with misic formulation
        self.do_one_case(one_case, X, 3, formulation="misic", float_type=np.float32)

    def test_diabetes_lightgbm_misic_safety_floor(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        lgbm_reg = lgb.sklearn.LGBMRegressor(n_estimators=5, max_depth=3)
        lgbm_reg.fit(X, y)
        one_case = {"predictor": lgbm_reg, "nonconvex": 0}

        # Test with misic formulation and safety_floor
        self.do_one_case(one_case, X, 3, formulation="misic", safety_floor=1e-5, float_type=np.float32)
