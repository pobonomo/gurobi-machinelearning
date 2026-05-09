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

"""Implementation of the Mišić (2020) formulation for tree ensembles."""

import numpy as np
from gurobipy import GRB

from ..base_predictor_constr import AbstractPredictorConstr
from ...exceptions import NoSolutionError


class MisicTreeEnsemble(AbstractPredictorConstr):
    """Formulate a tree ensemble using the Mišić (2020) formulation.

    This formulation uses split-indicator binary variables shared across all
    trees in the ensemble. It is often more efficient and numerically stable
    than modeling each tree independently.

    Reference:
        Mišić, V. V. (2020). Optimization of tree ensembles. Operations Research, 68(5), 1605-1624.
    """

    def __init__(
        self,
        gp_model,
        trees,
        input_vars,
        output_vars,
        epsilon=0.0,
        safety_floor=0.0,
        predictor=None,
        **kwargs,
    ):
        """Initialize the Mišić tree ensemble formulation.

        Parameters
        ----------
        gp_model : gurobipy.Model
            The Gurobi model to add the constraints to.
        trees : list of dict
            A list of flattened tree representations. Each dict should have:
            - 'children_left': array of left child indices
            - 'children_right': array of right child indices
            - 'feature': array of feature indices for splits
            - 'threshold': array of split thresholds
            - 'value': array of leaf values
            - 'n_features': total number of features
        input_vars : gurobipy.MVar
            The input variables.
        output_vars : gurobipy.MVar
            The output variables.
        epsilon : float, optional
            A small value to distinguish between <= and > splits.
        safety_floor : float, optional
            Thresholds with absolute value smaller than this will be clamped
            to this value to avoid numerical issues with Gurobi's tolerance.
        predictor : object, optional
            The original predictor object (e.g. LightGBM Booster) to use for
            error calculation.
        """
        self.trees = trees
        self.epsilon = epsilon
        self.safety_floor = safety_floor
        self.predictor = predictor
        self._default_name = "misic_tree"
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    def get_error(self, eps=None):
        if self._has_solution:
            if self.predictor is not None:
                # Use the original predictor for exact prediction
                in_val = self.input_values
                out_val = self.predictor.predict(in_val)
                r_val = np.abs(out_val.reshape(-1, 1) - self.output.X)
                return r_val
            # Fallback or simplified error check could be implemented here
            return np.zeros(self.output.shape)
        raise NoSolutionError()

    def _mip_model(self, **kwargs):

        model = self.gp_model
        _input = self.input
        output = self.output
        nex = _input.shape[0]
        n_features = self.trees[0]["n_features"]

        # 1. Extract unique thresholds per feature with optional clamping
        unique_thresholds = [set() for _ in range(n_features)]
        for tree in self.trees:
            not_leafs = tree["children_left"] >= 0
            features = tree["feature"][not_leafs]
            vals = tree["threshold"][not_leafs]
            for f, v in zip(features, vals):
                if 0 < abs(v) < self.safety_floor:
                    v = np.sign(v) * self.safety_floor
                unique_thresholds[f].add(v)

        sorted_thresholds = [sorted(list(s)) for s in unique_thresholds]

        # 2. Create shared binary variables z[j, k] = 1 iff x[j] <= threshold[j, k]
        z_vars = []
        for j, thresholds in enumerate(sorted_thresholds):
            if len(thresholds) > 0:
                z_j = model.addMVar(
                    (nex, len(thresholds)),
                    vtype=GRB.BINARY,
                    name=self._name_var(f"z_f{j}"),
                )
                z_vars.append(z_j)
                if len(thresholds) > 1:
                    model.addConstr(z_j[:, :-1] <= z_j[:, 1:])
            else:
                z_vars.append(None)

        # 3. Link input variables x[j] to shared binaries z[j, k]
        # We use Big-M formulation if bounds are available, otherwise indicators
        input_lb = _input.getAttr(GRB.Attr.LB)
        input_ub = _input.getAttr(GRB.Attr.UB)

        for j, thresholds in enumerate(sorted_thresholds):
            if len(thresholds) == 0:
                continue
            z_j = z_vars[j]
            x_j = _input[:, j]

            lb_j = input_lb[:, j]
            ub_j = input_ub[:, j]

            for k, val in enumerate(thresholds):
                # Big-M linking if possible
                if np.all(lb_j > -GRB.INFINITY) and np.all(ub_j < GRB.INFINITY):
                    # z[j, k] = 1 -> x[j] <= val
                    model.addConstr(x_j <= val + (ub_j - val) * (1 - z_j[:, k]))
                    # z[j, k] = 0 -> x[j] >= val + epsilon
                    # x[j] >= val + epsilon - (val + epsilon - lb_j) * z_j[:, k]
                    model.addConstr(
                        x_j
                        >= val + self.epsilon - (val + self.epsilon - lb_j) * z_j[:, k]
                    )
                else:
                    # Fallback to indicators
                    for i in range(nex):
                        model.addGenConstrIndicator(
                            z_j[i, k], 1, x_j[i], GRB.LESS_EQUAL, val
                        )
                        model.addGenConstrIndicator(
                            z_j[i, k], 0, x_j[i], GRB.GREATER_EQUAL, val + self.epsilon
                        )

        # 4. Model each tree using shared binaries
        tree_outputs = []
        for t, tree in enumerate(self.trees):
            children_left = tree["children_left"]
            children_right = tree["children_right"]
            features = tree["feature"]
            tree_thresholds = tree["threshold"]
            values = tree["value"]

            leaf_indices = np.where(children_left < 0)[0]
            y_t = model.addMVar(
                (nex, len(leaf_indices)),
                vtype=GRB.BINARY,
                name=self._name_var(f"y_t{t}"),
            )
            model.addConstr(y_t.sum(axis=1) == 1)
            leaf_map = {idx: i for i, idx in enumerate(leaf_indices)}

            # Traverse tree to build path constraints
            stack = [(0, [])] # (node, list of (z_jk, sense)) where sense is 1 for left, 0 for right
            while stack:
                u, path = stack.pop()
                left = children_left[u]
                if left >= 0:
                    right = children_right[u]
                    feat = features[u]
                    thresh = tree_thresholds[u]
                    # Need to account for clamping in lookup
                    if 0 < abs(thresh) < self.safety_floor:
                        thresh = np.sign(thresh) * self.safety_floor
                    
                    k = sorted_thresholds[feat].index(thresh)
                    z_jk = z_vars[feat][:, k]

                    stack.append((right, path + [(z_jk, 0)]))
                    stack.append((left, path + [(z_jk, 1)]))
                else:
                    l_idx = leaf_map[u]
                    y_ti_l = y_t[:, l_idx]
                    for z_jk, sense in path:
                        if sense == 1: # Left: x <= thresh => z_jk = 1
                            model.addConstr(y_ti_l <= z_jk)
                        else: # Right: x > thresh => z_jk = 0
                            model.addConstr(y_ti_l <= 1 - z_jk)

            tree_outputs.append(y_t @ values[leaf_indices, :])

        model.addConstr(output == sum(tree_outputs))

