import argparse
import cProfile
import pstats
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import gurobipy_pandas as gppd
import pandas as pd
from gurobi_ml import add_predictor_constr
from gurobi_ml.lightgbm import add_lgbm_booster_constr

from .data import setup_avocado, get_janos_data, setup_janos_model, setup_misic_model, setup_issue_496_model

def _execute_formulation(m, formulation, add_predictor_fn, extract_sol_fn, action):
    result = {'Formulation': formulation}
    
    if action == 'profile':
        pr = cProfile.Profile()
        pr.enable()
        add_predictor_fn()
        pr.disable()
        print(f"\n--- Profiling {formulation} ---")
        pstats.Stats(pr).sort_stats("tottime").print_stats(30)
        build_time = np.nan
    else:
        t0 = time.time()
        add_predictor_fn()
        build_time = time.time() - t0
        
    m.update()
    result.update({
        'Vars': m.NumVars,
        'Binaries': m.NumBinVars,
        'Constrs': m.NumConstrs,
        'Build (s)': build_time,
        'Status': np.nan,
        'Solve (s)': np.nan,
        'ObjVal': np.nan,
        'Gap (%)': np.nan,
        'MaxVio': np.nan
    })
    
    sol = None
    if action == 'solve':
        t0 = time.time()
        m.optimize()
        result['Solve (s)'] = time.time() - t0
        result['Status'] = 'Optimal' if m.Status == GRB.OPTIMAL else ('TimeLimit' if m.Status == GRB.TIME_LIMIT else f"Code {m.Status}")
            
        if hasattr(m, 'SolCount') and m.SolCount > 0:
            result['ObjVal'] = m.ObjVal
            try:
                result['Gap (%)'] = m.MIPGap * 100
            except AttributeError:
                pass
            result['MaxVio'] = max([m.BoundVio, m.ConstrVio, m.IntVio])
            sol = extract_sol_fn()
        elif m.Status == GRB.TIME_LIMIT:
            result['Status'] = 'TimeLimit (No Sol)'
            
    return result, sol

def crosscheck_sols(case_name, results, sols):
    print(f"\n--- Crosscheck: {case_name} ---")
    valid_sols = {k: v for k, v in sols.items() if v is not None}
    if not valid_sols:
        print("No optimal solutions to crosscheck.")
        return
        
    sol_df = pd.DataFrame({
        form: {f"{k}_{i}": val for k, v in sol.items() for i, val in enumerate(np.array(v).flatten())}
        for form, sol in valid_sols.items()
    })
    
    obj_series = pd.to_numeric(pd.DataFrame(results).set_index("Formulation").loc[list(valid_sols.keys()), 'ObjVal'], errors='coerce')
    obj_match = obj_series.max() - obj_series.min() <= 1e-4
    sol_match = (sol_df.max(axis=1) - sol_df.min(axis=1)).fillna(0) <= 1e-4
    
    if obj_match and sol_match.all():
        print("All formulations returned matching objective values and solutions (within 1e-4).")
    else:
        if not obj_match:
            print("MISMATCH in Objective Values:")
            print(obj_series.to_frame().to_markdown(floatfmt=".4f"))
        if not sol_match.all():
            print("MISMATCH in Solutions for following variables:")
            print(sol_df[~sol_match].to_markdown(floatfmt=".2e"))

def _run_case(name, formulations, action, timelimit, default_timelimit, verbose, build_base_fn):
    print(f"Running: {name} ...")
    results, sols = [], {}
    for formulation in formulations:
        m = gp.Model(name)
        m.Params.OutputFlag = 1 if verbose else 0
        m.Params.TimeLimit = timelimit if timelimit is not None else default_timelimit
        
        add_pred, ext_sol = build_base_fn(m, formulation)
        if add_pred is None: continue
            
        res, sol = _execute_formulation(m, formulation, add_pred, ext_sol, action)
        results.append(res)
        sols[formulation] = sol
        
    return results, sols

def run_avocado(formulations, action, timelimit, verbose):
    rf_reg, regions = setup_avocado()
    def build_base(m, formulation):
        price_vars = m.addMVar(len(regions), lb=0.5, ub=2.0, name="price")
        input_df = pd.DataFrame({"region": regions, "price": price_vars.tolist(), "year": 2022, "peak": 1})
        demand_vars = m.addMVar(len(regions), lb=-GRB.INFINITY, name="demand")
        m.setObjective(price_vars @ demand_vars, GRB.MAXIMIZE)
        
        def add_pred():
            add_predictor_constr(m, rf_reg, input_df, demand_vars, formulation=formulation, no_debug=True, epsilon=1e-5)
        return add_pred, lambda: {'price': price_vars.X, 'demand': demand_vars.X}
    return _run_case("Avocado Price Optimization", formulations, action, timelimit, 30, verbose, build_base)

def run_janos(name, n_estimators, max_depth, n_students, formulations, action, timelimit, verbose):
    historical_data, studentsdata_full = get_janos_data()
    features = ["merit", "SAT", "GPA"]
    regression = setup_janos_model(historical_data, features, n_estimators, max_depth)
    studentsdata = studentsdata_full.sample(n_students, random_state=1)
    
    def build_base(m, formulation):
        y_prob = gppd.add_vars(m, studentsdata, name='enroll_probability', lb=0, ub=1)
        merit_vars = gppd.add_vars(m, studentsdata, lb=0.0, ub=2.5, name='merit')
        
        input_data = studentsdata.copy()
        input_data['merit'] = merit_vars
        input_data = input_data[features]
        
        m.setObjective(y_prob.sum(), GRB.MAXIMIZE)
        m.addConstr(merit_vars.sum() <= 0.2 * n_students)
        
        def add_pred():
            add_predictor_constr(m, regression, input_data, y_prob, epsilon=1e-5, formulation=formulation)
        return add_pred, lambda: {'merit': merit_vars.gppd.X.values, 'enroll_prob': y_prob.gppd.X.values}
    return _run_case(f"Janos ({name}) - {n_estimators} trees, depth {max_depth}, {n_students} students", formulations, action, timelimit, 60, verbose, build_base)

def run_janos_cf(formulations, action, timelimit, verbose):
    historical_data, studentsdata_full = get_janos_data()
    features = ["SAT", "GPA"]
    regression = setup_janos_model(historical_data, features, 100, 10)
    student = studentsdata_full.iloc[[0]].copy()
    
    def build_base(m, formulation):
        x_vars = {f: m.addVar(lb=historical_data[f].min(), ub=historical_data[f].max(), name=f"x_{f}") for f in features}
        y = m.addVar(lb=0.8, ub=1, name="y")
        
        diff = m.addVars(features, lb=0)
        for f in features:
            orig_val = student.loc[student.index[0], f]
            m.addConstr(diff[f] >= x_vars[f] - orig_val)
            m.addConstr(diff[f] >= orig_val - x_vars[f])
        
        m.setObjective(diff.sum(), GRB.MINIMIZE)
        
        def add_pred():
            add_predictor_constr(m, regression, pd.DataFrame([x_vars]), y, epsilon=1e-5, formulation=formulation)
        return add_pred, lambda: {'x': [x_vars[f].X for f in features], 'y': y.X, 'diff': [diff[f].X for f in features]}
    return _run_case("Janos Counterfactual", formulations, action, timelimit, 60, verbose, build_base)

def run_lgbm_synthetic(formulations, action, timelimit, verbose):
    n_trees, max_depth = 100, 6
    model, n_features = setup_misic_model(n_trees, max_depth)
    
    def build_base(m, formulation):
        if formulation not in ["leaf", "misic"]: return None, None
        
        x_vars = m.addMVar(shape=n_features, lb=-5.0, ub=5.0, name='x')
        y_var = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='score')
        m.setObjective(y_var, GRB.MAXIMIZE)
        
        def add_pred():
            add_lgbm_booster_constr(m, model, x_vars, y_var, formulation=formulation)
        return add_pred, lambda: {'x': x_vars.X, 'score': y_var.X}
    return _run_case("LightGBM Synthetic", formulations, action, timelimit, 60, verbose, build_base)

def run_issue_496(formulations, action, timelimit, verbose):
    model, n_features = setup_issue_496_model()
    
    def build_base(m, formulation):
        if formulation not in ["leaf", "misic"]: return None, None
        
        x_vars = m.addMVar(shape=n_features, lb=-10.0, ub=10.0, name='x')
        y_var = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='score')
        
        x_test = [1e-10, 0.0, 0.0, 0.0, 0.0]
        for j in range(n_features):
            x_vars[j].LB = x_vars[j].UB = x_test[j]
            
        m.setObjective(y_var, GRB.MAXIMIZE)
        
        def add_pred():
            kwargs = {"safety_floor": 1e-5} if formulation == "misic" else {}
            add_lgbm_booster_constr(m, model, x_vars, y_var, formulation=formulation, **kwargs)
        return add_pred, lambda: {'score': y_var.X}
    return _run_case("LightGBM Issue 496", formulations, action, timelimit, 60, verbose, build_base)

BENCHMARKS = {
    "avocado": run_avocado,
    "janos": lambda f, a, t, v: run_janos("Standard", 100, 6, 100, f, a, t, v),
    "janos-small": lambda f, a, t, v: run_janos("Small", 50, 6, 10, f, a, t, v),
    "janos-deep": lambda f, a, t, v: run_janos("Deep", 10, 12, 5, f, a, t, v),
    "janos-large": lambda f, a, t, v: run_janos("Large", 50, 6, 200, f, a, t, v),
    "janos-cf": run_janos_cf,
    "lgbm-synthetic": run_lgbm_synthetic,
    "issue_496": run_issue_496,
}

def main():
    parser = argparse.ArgumentParser(description="Gurobi Machine Learning Benchmarks")
    parser.add_argument("action", choices=["solve", "build", "profile"], help="Action to perform: solve (builds + optimizes), build (only builds model), profile (profiles the build step)")
    parser.add_argument("cases", nargs="+", choices=["all"] + list(BENCHMARKS.keys()), help="Cases to run")
    parser.add_argument("--formulations", type=str, nargs="+", default=["leaf", "misic", "vidal"], help="Which formulations to test")
    parser.add_argument("--timelimit", type=int, default=None, help="Time limit for the solver in seconds (overrides case defaults)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose Gurobi solver output")
    
    args = parser.parse_args()
    runs = list(BENCHMARKS.keys()) if "all" in args.cases else args.cases
    all_results, all_sols = [], {}
    
    for r in runs:
        case_results, case_sols = BENCHMARKS[r](args.formulations, args.action, args.timelimit, args.verbose)
        all_results.extend({'Case': r, **res} for res in case_results)
        all_sols[r] = case_sols
        
    print("\n" + "="*80 + f"\nBENCHMARK RESULTS (Action: {args.action})\n" + "="*80)
    
    df = pd.DataFrame(all_results).set_index(["Case", "Formulation"]).dropna(axis=1, how='all')
    df[df.columns.intersection(['Vars', 'Binaries', 'Constrs'])] = df[df.columns.intersection(['Vars', 'Binaries', 'Constrs'])].astype('Int64')
    df = df.round({k: v for k, v in {'Build (s)': 2, 'Solve (s)': 2, 'ObjVal': 4, 'Gap (%)': 2}.items() if k in df.columns})
        
    print(df.to_markdown())
    
    if args.action == 'solve':
        for r in runs:
            crosscheck_sols(r, [res for res in all_results if res['Case'] == r], all_sols[r])

if __name__ == "__main__":
    main()
