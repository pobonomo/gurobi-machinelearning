# Copyright © 2023 Gurobi Optimization, LLC
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

"""Module for formulating a :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
in a :gurobipy:`model`.
"""

import gurobipy as gp
import numpy as np

from ..exceptions import NoModel
from ..modeling.base_predictor_constr import AbstractPredictorConstr
from ..modeling.get_convertor import get_convertor
from ..register_user_predictor import user_predictors
from ..xgboost_sklearn_api import xgboost_sklearn_convertors
from .predictors_list import sklearn_predictors
from .skgetter import SKgetter


def add_transformed_target_regressor_constr(
    gp_model, transformed_target_regressor, input_vars, output_vars=None, **kwargs
):
    """Formulate transformed_target_regressor into gp_model.

    The formulation predicts the values of output_vars using input_vars according to
    transformed_target_regressor.

    Parameters
    ----------
    gp_model : :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    transformed_target_regressor : :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
        The pipeline to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for regression in model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for regression in model.

    Returns
    -------
    PipelineConstr
        Object containing information about what was added to gp_model to embed the
        predictor into it

    Raises
    ------
    NoModel
        If the translation to Gurobi of one of the elements in the transformed_target_regressor
        is not implemented or recognized.

    Notes
    -----
    |VariablesDimensionsWarn|
    """
    return TransformedTargetRegressorConstr(
        gp_model, transformed_target_regressor, input_vars, output_vars, **kwargs
    )


class TransformedTargetRegressorConstr(SKgetter, AbstractPredictorConstr):
    """Class to formulate a trained :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
    in a gurobipy model.

    |ClassShort|
    """

    def __init__(
        self,
        gp_model,
        transformed_target_regressor,
        input_vars,
        output_vars=None,
        **kwargs,
    ):
        self._regression = None
        self._default_name = "transformed_target_regressor"
        SKgetter.__init__(self, transformed_target_regressor, input_vars, **kwargs)
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, validate_input=False, **kwargs
        )

    def _build_submodel(self, gp_model, *args, **kwargs):
        """Predict output from input using predictor or transformer.

        Pipelines are different from other objects because they can't validate
        their input and output. They are just containers of other objects that will
        do it.
        """
        self._mip_model(**kwargs)
        assert self.output is not None
        assert self.input is not None
        # We can call validate only after the model is created
        self._validate()
        return self

    def _mip_model(self, **kwargs):
        transformed_target_regressor = self.predictor
        gp_model = self.gp_model
        input_vars = self.input
        kwargs["validate_input"] = True

        predictor = transformed_target_regressor.regression_
        predictors = sklearn_predictors() | user_predictors()
        predictors |= xgboost_sklearn_convertors()
        convertor = get_convertor(predictor, predictors)
        if convertor is None:
            raise NoModel(
                self.predictor,
                f"I don't know how to deal with that object: {predictor}",
            )

        regression_constr = convertor(
            gp_model, predictor, input_vars, output_vars=None, **kwargs
        )
        if self.output is None:
            self.output = gp_model.AddMVar(
                regression_constr.output.shape, lb=-gp.GRB.INFINITY
            )
        if transformed_target_regressor.inverse_func == np.exp:
            for index in np.ndindex(self.output.shape):
                gp_model.AddGenConstrExp(
                    self.output[index], regression_constr.output[index]
                )

    def print_stats(self, file=None):
        """Print statistics on model additions stored by this class.

        This function prints detailed statistics on the variables
        and constraints that where added to the model.

        The transformed_target_regressor version includes a summary of the steps that it contains.

        Parameters
        ----------

        file: None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """
        super().print_stats(file=file)
        print(file=file)

        self._print_container_steps("Step", self._steps, file=file)
