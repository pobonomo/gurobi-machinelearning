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
            return np.zeros(self.output.shape)
        raise NoSolutionError()

    def _mip_model(self, **kwargs):
        model = self.gp_model
        _input = self.input
        output = self.output
        nex = _input.shape[0]
        n_features = self.trees[0]["n_features"]

        # 1. Extract unique thresholds and pre-calculate offsets
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
        n_splits_per_feat = np.array([len(s) for s in sorted_thresholds])
        total_splits = np.sum(n_splits_per_feat)
        split_offsets = np.cumsum(np.concatenate(([0], n_splits_per_feat)))
        threshold_to_idx = [
            {val: k for k, val in enumerate(thresholds)}
            for thresholds in sorted_thresholds
        ]

        # 2. Shared binaries Z: (nex, total_splits)
        if total_splits > 0:
            z_vars = model.addMVar(
                (nex, total_splits), vtype=GRB.BINARY, name=self._name_var("z")
            )
            for j in range(n_features):
                if n_splits_per_feat[j] > 1:
                    off = split_offsets[j]
                    model.addConstr(
                        z_vars[:, off : off + n_splits_per_feat[j] - 1]
                        <= z_vars[:, off + 1 : off + n_splits_per_feat[j]]
                    )
        else:
            z_vars = None

        # 3. Link input variables x[j] to shared binaries z[j, k]
        input_lb = _input.getAttr(GRB.Attr.LB)
        input_ub = _input.getAttr(GRB.Attr.UB)

        for j in range(n_features):
            thresholds = sorted_thresholds[j]
            if not thresholds:
                continue
            
            off = split_offsets[j]
            z_j = z_vars[:, off : off + len(thresholds)]
            x_j = _input[:, j]

            lb_j = input_lb[:, j]
            ub_j = input_ub[:, j]
            vals = np.array(thresholds)

            if np.all(lb_j > -GRB.INFINITY) and np.all(ub_j < GRB.INFINITY):
                model.addConstr(
                    x_j[:, np.newaxis] <= vals + (ub_j[:, np.newaxis] - vals) * (1 - z_j)
                )
                model.addConstr(
                    x_j[:, np.newaxis] >= vals + self.epsilon - (vals + self.epsilon - lb_j[:, np.newaxis]) * z_j
                )
            else:
                for k, val in enumerate(thresholds):
                    for i in range(nex):
                        model.addGenConstrIndicator(
                            z_j[i, k], 1, x_j[i], GRB.LESS_EQUAL, val
                        )
                        model.addGenConstrIndicator(
                            z_j[i, k], 0, x_j[i], GRB.GREATER_EQUAL, val + self.epsilon
                        )

        # 4. Model all trees using shared binaries
        tree_leaf_counts = np.array([
            len(np.where(tree["children_left"] < 0)[0]) for tree in self.trees
        ])
        total_leaves = np.sum(tree_leaf_counts)
        leaf_offsets = np.cumsum(np.concatenate(([0], tree_leaf_counts)))

        y_leaf = model.addMVar(
            (nex, total_leaves), vtype=GRB.BINARY, name=""
        )

        for t in range(len(self.trees)):
            off = leaf_offsets[t]
            model.addConstr(y_leaf[:, off : off + tree_leaf_counts[t]].sum(axis=1) == 1)

        # Collect path constraints for vectorization
        all_l_left = []
        all_z_left = []
        all_l_right = []
        all_z_right = []

        for t, tree in enumerate(self.trees):
            off = leaf_offsets[t]
            children_left = tree["children_left"]
            children_right = tree["children_right"]
            features = tree["feature"]
            tree_thresholds = tree["threshold"]

            leaf_indices = np.where(children_left < 0)[0]
            leaf_map = {node_idx: i for i, node_idx in enumerate(leaf_indices)}

            stack = [(0, [])]
            while stack:
                u, path = stack.pop()
                left = children_left[u]
                if left >= 0:
                    right = children_right[u]
                    feat = features[u]
                    thresh = tree_thresholds[u]
                    if 0 < abs(thresh) < self.safety_floor:
                        thresh = np.sign(thresh) * self.safety_floor
                    k = threshold_to_idx[feat][thresh]
                    z_idx = split_offsets[feat] + k

                    stack.append((right, path + [(z_idx, 0)]))
                    stack.append((left, path + [(z_idx, 1)]))
                else:
                    l_idx = off + leaf_map[u]
                    for z_idx, sense in path:
                        if sense == 1:
                            all_l_left.append(l_idx)
                            all_z_left.append(z_idx)
                        else:
                            all_l_right.append(l_idx)
                            all_z_right.append(z_idx)

        # Vectorized addition of path constraints
        if all_l_left:
            model.addConstr(y_leaf[:, all_l_left] <= z_vars[:, all_z_left])
        if all_l_right:
            model.addConstr(y_leaf[:, all_l_right] <= 1 - z_vars[:, all_z_right])

        # 5. Output calculation
        tree_outputs = []
        for t, tree in enumerate(self.trees):
            off = leaf_offsets[t]
            leaf_indices = np.where(tree["children_left"] < 0)[0]
            values = tree["value"]
            tree_outputs.append(
                y_leaf[:, off : off + tree_leaf_counts[t]] @ values[leaf_indices, :]
            )

        model.addConstr(output == sum(tree_outputs))
