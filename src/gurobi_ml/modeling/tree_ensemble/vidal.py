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

"""Implementation of the Vidal (2021) flow-based formulation for tree ensembles."""

import numpy as np
from gurobipy import GRB

from ..base_predictor_constr import AbstractPredictorConstr
from ...exceptions import NoSolutionError


class VidalTreeEnsemble(AbstractPredictorConstr):
    """Formulate a tree ensemble using the Vidal (2021) flow-based formulation.

    This formulation models each tree as a network flow problem. It uses
    continuous flow variables for each node/edge and links them to shared
    split-indicator binary variables. It often provides a tighter LP relaxation
    than leaf-based formulations.

    Reference:
        Parmentier, A., & Vidal, T. (2021). Optimal counterfactual explanations
        in tree ensembles. ICML.
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
        """Initialize the Vidal tree ensemble formulation.

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
            The original predictor object to use for error calculation.
        """
        self.trees = trees
        self.epsilon = epsilon
        self.safety_floor = safety_floor
        self.predictor = predictor
        self._default_name = "vidal_tree"
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

        # Build a fast mapping for thresholds
        threshold_to_idx = [
            {val: k for k, val in enumerate(thresholds)}
            for thresholds in sorted_thresholds
        ]

        # 2. Shared binaries Z: (nex, total_splits)
        if total_splits > 0:
            z_vars = model.addMVar(
                (nex, total_splits), vtype=GRB.BINARY, name=self._name_var("z")
            )
            # Monotonicity: z_j,k <= z_j,k+1
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
                    x_j[:, np.newaxis]
                    <= vals + (ub_j[:, np.newaxis] - vals) * (1 - z_j)
                )
                model.addConstr(
                    x_j[:, np.newaxis]
                    >= vals
                    + self.epsilon
                    - (vals + self.epsilon - lb_j[:, np.newaxis]) * z_j
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

        # 4. Model all trees using flow formulation
        tree_node_counts = np.array([len(tree["children_left"]) for tree in self.trees])
        total_nodes = np.sum(tree_node_counts)
        node_offsets = np.cumsum(np.concatenate(([0], tree_node_counts)))

        y_node = model.addMVar((nex, total_nodes), lb=0.0, ub=1.0, name="")

        # Reachability pruning: set flow to 0 if node is unreachable based on input bounds
        from ..decision_tree.decision_tree_model import _compute_leafs_bounds

        feature_is_fixed = (_input.lb == _input.ub).all(axis=0)
        input_lb = _input.getAttr(GRB.Attr.LB)
        input_ub = _input.getAttr(GRB.Attr.UB)

        for t, tree in enumerate(self.trees):
            off = node_offsets[t]
            node_lb, node_ub = _compute_leafs_bounds(
                tree, feature_is_fixed, self.epsilon
            )

            # node_lb shape: (n_features, node_count)
            # input_ub shape: (nex, n_features)
            # reachable if: for all features, input_ub >= node_lb and input_lb <= node_ub

            # Vectorized reachability check for this tree
            # We want (nex, node_count) boolean mask
            # For each node k: reachable[i, k] = all(input_ub[i, f] >= node_lb[f, k] and input_lb[i, f] <= node_ub[f, k])

            # Using numpy broadcasting:
            # input_ub[:, :, np.newaxis] is (nex, n_features, 1)
            # node_lb[np.newaxis, :, :] is (1, n_features, node_count)
            is_reachable = np.all(
                (input_ub[:, :, np.newaxis] >= node_lb[np.newaxis, :, :])
                & (input_lb[:, :, np.newaxis] <= node_ub[np.newaxis, :, :]),
                axis=1,
            )

            # Set UB to 0 for unreachable nodes
            y_node_tree = y_node[:, off : off + tree["capacity"]]
            y_node_tree[~is_reachable].setAttr(GRB.Attr.UB, 0.0)

        # Root flow = 1
        model.addConstr(y_node[:, node_offsets[:-1]] == 1.0)

        # Vectorized data collection
        all_children_left = np.concatenate([t["children_left"] for t in self.trees])
        all_children_right = np.concatenate([t["children_right"] for t in self.trees])
        all_features = np.concatenate([t["feature"] for t in self.trees])
        all_thresholds = np.concatenate([t["threshold"] for t in self.trees])

        # Adjust thresholds for safety floor
        mask = (np.abs(all_thresholds) > 0) & (
            np.abs(all_thresholds) < self.safety_floor
        )
        all_thresholds[mask] = np.sign(all_thresholds[mask]) * self.safety_floor

        is_internal = all_children_left >= 0
        internal_indices = np.where(is_internal)[0]

        if len(internal_indices) > 0:
            # Tree-specific offsets for nodes
            tree_ids = np.repeat(np.arange(len(self.trees)), tree_node_counts)
            node_offsets_expanded = node_offsets[tree_ids]

            all_internal = internal_indices
            all_lefts = (
                all_children_left[internal_indices]
                + node_offsets_expanded[internal_indices]
            )
            all_rights = (
                all_children_right[internal_indices]
                + node_offsets_expanded[internal_indices]
            )

            # Map features and thresholds to shared binary indices
            feats = all_features[internal_indices]
            threshs = all_thresholds[internal_indices]

            # This part still needs a loop or a smarter map if possible,
            # but it's only over total splits which is manageable.
            all_z_indices = np.zeros(len(internal_indices), dtype=int)
            for i, (f, v) in enumerate(zip(feats, threshs)):
                all_z_indices[i] = split_offsets[f] + threshold_to_idx[f][v]

            # Flow conservation: y_u = y_l + y_r
            model.addConstr(
                y_node[:, all_internal] == y_node[:, all_lefts] + y_node[:, all_rights]
            )
            # Binary linking: y_l <= z_jk, y_r <= 1 - z_jk
            z_linked = z_vars[:, all_z_indices]
            model.addConstr(y_node[:, all_lefts] <= z_linked)
            model.addConstr(y_node[:, all_rights] <= 1 - z_linked)

        # 5. Output calculation (Vectorized across trees)
        is_leaf = all_children_left < 0
        leaf_indices = np.where(is_leaf)[0]

        # We can't easily vectorize the whole sum if leaf values are different
        # shapes but here they should be consistent.
        # Actually, let's keep the per-tree loop for the final output sum
        # to handle potential multi-output or different tree values.
        tree_outputs = []
        for t, tree in enumerate(self.trees):
            off = node_offsets[t]
            t_leafs = np.where(tree["children_left"] < 0)[0]
            values = tree["value"]
            tree_outputs.append(y_node[:, off + t_leafs] @ values[t_leafs, :])

        model.addConstr(output == sum(tree_outputs))
