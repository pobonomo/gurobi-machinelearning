# Copyright © 2023-2026 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for formulating a
:external+sklearn:py:class:`sklearn.ensemble.GradientBoostingRegressor`
into a :external+gurobi:py:class:`Model`.
"""

from gurobipy import GRB

from ..modeling import AbstractPredictorConstr
from ..modeling.tree_ensemble.misic import MisicTreeEnsemble
from .decision_tree_regressor import add_decision_tree_regressor_constr
from .skgetter import SKgetter


def add_gradient_boosting_regressor_constr(
    gp_model,
    gradient_boosting_regressor,
    input_vars,
    output_vars=None,
    formulation="leaf",
    **kwargs,
):
    """Formulate gradient_boosting_regressor into gp_model.

    The formulation predicts the values of output_vars using input_vars
    according to gradient_boosting_regressor. See our :ref:`User's Guide
    <Gradient Boosting Regression>` for details on the mip formulation used.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    gradient_boosting_regressor : :external+sklearn:py:class:`sklearn.ensemble.GradientBoostingRegressor`
        The gradient boosting regressor to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for gradient boosting regressor in model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for gradient boosting regressor in model.
    formulation : str, optional
        The formulation to use. One of "leaf" or "misic". Default is "leaf".
    safety_floor : float, optional
        Thresholds with absolute value smaller than this will be clamped
        to this value to avoid numerical issues with Gurobi's tolerance.
        Only used if formulation is "misic".

    Returns
    -------
    GradientBoostingRegressorConstr
        Object containing information about what was added to gp_model to formulate
        gradient_boosting_regressor.

    Notes
    -----
    |VariablesDimensionsWarn|

    Also see
    :py:func:`gurobi_ml.sklearn.decision_tree_regressor.add_decision_tree_regressor`
    for specific parameters to model decision tree estimators.
    """
    return GradientBoostingRegressorConstr(
        gp_model,
        gradient_boosting_regressor,
        input_vars,
        output_vars,
        formulation=formulation,
        **kwargs,
    )


class GradientBoostingRegressorConstr(SKgetter, AbstractPredictorConstr):
    """Class to formulate a trained
    :external+sklearn:py:class:`sklearn.ensemble.GradientBoostingRegressor`
    in a gurobipy model.

    |ClassShort|
    """

    def __init__(
        self,
        gp_model,
        predictor,
        input_vars,
        output_vars,
        formulation="leaf",
        safety_floor=0.0,
        **kwargs,
    ):
        self._output_shape = 1
        self.estimators_ = []
        self._default_name = "gbtree_reg"
        self.formulation = formulation
        self.safety_floor = safety_floor
        SKgetter.__init__(self, predictor, input_vars)
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Predict output variables y from input variables X using the
        decision tree.

        Both X and y should be array or list of variables of conforming dimensions.
        """
        model = self.gp_model
        predictor = self.predictor

        _input = self._input
        output = self._output
        nex = _input.shape[0]

        outdim = output.shape[1]
        assert outdim == 1, (
            "Output dimension of gradient boosting regressor should be 1"
        )

        if self.formulation == "misic":
            trees = []
            for i in range(predictor.n_estimators_):
                tree = predictor.estimators_[i][0].tree_
                trees.append(
                    {
                        "capacity": tree.node_count,
                        "children_left": tree.children_left,
                        "children_right": tree.children_right,
                        "feature": tree.feature,
                        "threshold": tree.threshold,
                        "value": tree.value[:, 0, :],
                        "n_features": predictor.n_features_in_,
                    }
                )
            sum_trees = model.addMVar(
                output.shape, lb=-GRB.INFINITY, name=self._name_var("sum_trees")
            )
            self.estimators_ = [
                MisicTreeEnsemble(
                    model,
                    trees,
                    _input,
                    sum_trees,
                    safety_floor=self.safety_floor,
                    predictor=predictor,
                    **kwargs,
                )
            ]
            constant = predictor.init_.constant_
            model.addConstr(
                output == predictor.learning_rate * sum_trees + constant[0][0]
            )
            return

        estimators = []
        if self._no_debug:
            kwargs["no_record"] = True

        tree_vars = model.addMVar(
            (nex, predictor.n_estimators_, 1),
            lb=-GRB.INFINITY,
            name=self._name_var("esimator"),
        )

        for i in range(predictor.n_estimators_):
            tree = predictor.estimators_[i]
            if self.verbose:
                self._timer.timing(f"Estimator {i}")
            estimators.append(
                add_decision_tree_regressor_constr(
                    model, tree[0], _input, tree_vars[:, i, :], **kwargs
                )
            )
        self.estimators_ = estimators

        constant = predictor.init_.constant_
        model.addConstr(
            output == predictor.learning_rate * tree_vars.sum(axis=1) + constant[0][0]
        )

    def print_stats(self, abbrev=False, file=None):
        """Print statistics on model additions stored by this class.

        This function prints detailed statistics on the variables
        and constraints that where added to the model.

        Includes a summary of the estimators that it contains.

        Parameters
        ----------

        file: None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """
        super().print_stats(abbrev=abbrev, file=file)
        if abbrev or self._no_debug:
            return
        print(file=file)

        self._print_container_steps("Estimator", self.estimators_, file=file)
