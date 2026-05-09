import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import numpy as np
from sklearn import datasets
from ..fixed_formulation import FixedRegressionModel

class TestSklearnEnsembleMisic(FixedRegressionModel):
    """Test the Mišić formulation for Scikit-Learn ensembles."""

    def test_diabetes_gbr_misic(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        gbr = GradientBoostingRegressor(n_estimators=5, max_depth=3)
        gbr.fit(X, y)
        one_case = {"predictor": gbr, "nonconvex": 0}

        # Test with misic formulation
        self.do_one_case(one_case, X, 3, formulation="misic", float_type=np.float32)

    def test_diabetes_rf_misic(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        rf = RandomForestRegressor(n_estimators=5, max_depth=3)
        rf.fit(X, y)
        one_case = {"predictor": rf, "nonconvex": 0}

        # Test with misic formulation
        self.do_one_case(one_case, X, 3, formulation="misic", float_type=np.float32)
