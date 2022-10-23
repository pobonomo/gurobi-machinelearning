# Copyright © 2022 Gurobi Optimization, LLC
""" Module for inserting simple Scikit-Learn regression models into a gurobipy model

All linear models should work:
   - :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
   - :external+sklearn:py:class:`sklearn.linear_model.Ridge`
   - :external+sklearn:py:class:`sklearn.linear_model.Lasso`

Also does :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression`
"""

import numpy as np

from ..modeling import AbstractPredictorConstr
from .skgetter import SKgetter


def _name(index, name):
    index = f"{index}".replace(" ", "")
    return f"{name}[{index}]"


class BaseSKlearnRegressionConstr(SKgetter, AbstractPredictorConstr):
    """Predict a Gurobi variable using a Linear Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, predictor, input_vars, output_vars=None, **kwargs):
        SKgetter.__init__(self, predictor)
        AbstractPredictorConstr.__init__(
            self,
            grbmodel,
            input_vars,
            output_vars,
            **kwargs,
        )

    def add_regression_constr(self):
        """Add the prediction constraints to Gurobi"""
        coefs = self.predictor.coef_.T
        intercept = self.predictor.intercept_
        self.model.addConstr(self._output == self._input @ coefs + intercept, name="linreg")

    def print_stats(self, file=None):
        """Print statistics about submodel created"""
        super().print_stats(file)


class LinearRegressionConstr(BaseSKlearnRegressionConstr):
    """Predict a Gurobi variable using a linear regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, predictor, input_vars, output_vars=None, **kwargs):
        BaseSKlearnRegressionConstr.__init__(
            self,
            grbmodel,
            predictor,
            input_vars,
            output_vars,
            **kwargs,
        )

    def _mip_model(self):
        """Add the prediction constraints to Gurobi"""
        self.add_regression_constr()


class LogisticRegressionConstr(BaseSKlearnRegressionConstr):
    """Predict a Gurobi variable using a logistic regression that
    takes another Gurobi matrix variable as input.

    """

    def __init__(self, grbmodel, predictor, input_vars, output_vars=None, pwl_attributes=None, **kwargs):
        if pwl_attributes is None:
            self.attributes = self.default_pwl_attributes()
        else:
            self.attributes = pwl_attributes

        BaseSKlearnRegressionConstr.__init__(
            self,
            grbmodel,
            predictor,
            input_vars,
            output_vars,
            **kwargs,
        )

    @staticmethod
    def default_pwl_attributes():
        """Default attributes for approximating the logistic function with Gurobi

        See `Gurobi's User Manual <https://www.gurobi.com/documentation/9.1/refman/general_constraint_attribu.html>`_
        for the meaning of the attributes.
        """
        return {"FuncPieces": -1, "FuncPieceLength": 0.01, "FuncPieceError": 0.01, "FuncPieceRatio": -1.0}

    def _mip_model(self):
        """Add the prediction constraints to Gurobi"""
        outputvars = self._output
        self._create_output_vars(self._input, name="affine_trans")
        affinevars = self._output
        self.add_regression_constr()
        self._output = outputvars
        for index in np.ndindex(outputvars.shape):
            gc = self.model.addGenConstrLogistic(
                affinevars[index],
                outputvars[index],
                name=_name(index, "logistic"),
            )
        numgc = self.model.NumGenConstrs
        self.model.update()
        for gc in self.model.getGenConstrs()[numgc:]:
            for attr, val in self.attributes.items():
                gc.setAttr(attr, val)


def add_linear_regression_constr(model, linear_regression, input_vars, output_vars=None, **kwargs):
    """Use `linear_regression` to predict the value of `output_vars` using `input_vars` in `model`

    Parameters
    ----------
    model: `gp.Model <https://www.gurobi.com/documentation/current/refman/py_model.html>`_
        The gurobipy model where the predictor should be inserted.
    linear_regression: : external+sklearn: py: class: `sklearn.linear_model.LinearRegression`
     The linear regression to insert. It can be of any of the following types:
         * : external+sklearn: py: class: `sklearn.linear_model.LinearRegression`
         * : external+sklearn: py: class: `sklearn.linear_model.Ridge`
         * : external+sklearn: py: class: `sklearn.linear_model.Lasso`
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    Returns
    -------
    LinearRegressionConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Note
    ----
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars
    """
    return LinearRegressionConstr(model, linear_regression, input_vars, output_vars, **kwargs)


def add_logistic_regression_constr(model, logistic_regression, input_vars, output_vars=None, pwl_attributes=None, **kwargs):
    """Use `logistic_regression` to predict the value of `output_vars` using `input_vars` in `model`

    Parameters
    ----------
    model: `gp.Model <https://www.gurobi.com/documentation/current/refman/py_model.html>`_
        The gurobipy model where the predictor should be inserted.
    logistic_regression: :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression`
        The logistic regression to insert.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.
    pwl_attributes: dict, optional
        Dictionary for non-default attributes for Gurobi to build the piecewise linear
        approximation of the logistic function.
        The default values for those attributes set in the package can be obtained
        with LogisticRegressionConstr.default_pwl_attributes().
        The dictionary keys should be the `attributes for modeling piece wise linear functions
        <https://www.gurobi.com/documentation/9.1/refman/general_constraint_attribu.html>`_
        and the values the corresponding value the users wants to pass to Gurobi.

    Returns
    -------
    LogisticRegressionConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Note
    ----
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars
    """
    return LogisticRegressionConstr(
        model, logistic_regression, input_vars, output_vars, pwl_attributes=pwl_attributes, **kwargs
    )
