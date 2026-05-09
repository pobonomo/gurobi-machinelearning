import gurobipy as gp
import gurobipy_pandas as gppd
import numpy as np
import pandas as pd
import time
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from gurobi_ml import add_predictor_constr

def benchmark_formulations(name, model_fit, create_optimization_problem):
    results = []
    formulations = ["leaf", "misic", "vidal"]
    
    print(f"\nBenchmarking Case: {name}")
    print("=" * 105)
    print(f"{'Formulation':<12} | {'Vars':<8} | {'Binaries':<10} | {'Constrs':<10} | {'Build (s)':<10} | {'Solve (s)':<10} | {'Gap':<8}")
    print("-" * 105)

    for formulation in formulations:
        m = gp.Model(f"{name}_{formulation}")
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = 30
        
        # Create optimization problem structure
        input_data, output_vars, extra_params = create_optimization_problem(m)
        
        start_build = time.time()
        add_predictor_constr(m, model_fit, input_data, output_vars, formulation=formulation, no_debug=True, **extra_params)
        end_build = time.time()
        
        # Optimization
        m.update()
        n_vars = m.NumVars
        n_constrs = m.NumConstrs
        n_binaries = m.NumBinVars
        
        start_solve = time.time()
        m.optimize()
        end_solve = time.time()
        
        solve_time = end_solve - start_solve
        build_time = end_build - start_build
        
        gap = m.MIPGap if hasattr(m, "MIPGap") and m.Status == gp.GRB.TIME_LIMIT else 0
        if m.Status != gp.GRB.OPTIMAL and m.Status != gp.GRB.TIME_LIMIT:
            solve_time = -1.0 # Failed
            
        print(f"{formulation:<12} | {n_vars:<8} | {n_binaries:<10} | {n_constrs:<10} | {build_time:<10.2f} | {solve_time:<10.2f} | {gap:<8.2%}")
        
        results.append({
            "Case": name,
            "Formulation": formulation,
            "Vars": n_vars,
            "Binaries": n_binaries,
            "Constrs": n_constrs,
            "Build": build_time,
            "Solve": solve_time,
            "Gap": gap
        })
    return results

# --- Case 1: Avocado Price Optimization ---
def setup_avocado():
    data_url = "https://raw.githubusercontent.com/Gurobi/modeling-examples/master/price_optimization/"
    avocado = pd.read_csv(data_url + "HABdata_2019_2022.csv")
    avocado_old = pd.read_csv(data_url + "kaggledata_till2018.csv")
    avocado["date"] = pd.to_datetime(avocado["date"], format="%m/%d/%y %H:%M")
    avocado_old["date"] = pd.to_datetime(avocado_old["date"], format="%m/%d/%y")
    avocado = pd.concat([avocado, avocado_old])
    avocado["year"] = pd.DatetimeIndex(avocado["date"]).year
    avocado["month"] = pd.DatetimeIndex(avocado["date"]).month
    peak_months = range(2, 8)
    avocado["peak"] = avocado["month"].apply(lambda x: 1 if x in peak_months else 0)
    avocado["units_sold"] = avocado["units_sold"] / 1000000
    avocado = avocado[avocado["type"] == "Conventional"]
    regions = ["Great_Lakes", "Midsouth", "Northeast", "West"]
    df = avocado[avocado.region.isin(regions)]
    
    X = df[["region", "price", "year", "peak"]]
    y = df["units_sold"]
    feat_transform = make_column_transformer(
        (OneHotEncoder(drop="first"), ["region"]),
        (StandardScaler(), ["price", "year"]),
        ("passthrough", ["peak"]),
        verbose_feature_names_out=False,
        remainder="drop",
    )
    rf_reg = make_pipeline(feat_transform, RandomForestRegressor(n_estimators=100, max_depth=8, random_state=1))
    rf_reg.fit(X, y)

    def create_optimization_problem(m):
        price_vars = m.addMVar(len(regions), lb=0.5, ub=2.0, name="price")
        input_df = pd.DataFrame({
            "region": regions,
            "price": price_vars.tolist(),
            "year": [2022] * len(regions),
            "peak": [1] * len(regions)
        })
        demand_vars = m.addMVar(len(regions), lb=-gp.GRB.INFINITY, name="demand")
        m.setObjective(price_vars @ demand_vars, gp.GRB.MAXIMIZE)
        return input_df, demand_vars, {"epsilon": 1e-5}

    return rf_reg, create_optimization_problem

# --- Case 2: Janos Student Admission ---
def setup_janos():
    janos_data_url = "https://raw.githubusercontent.com/INFORMSJoC/2020.1023/master/data/"
    historical_data = pd.read_csv(janos_data_url + "college_student_enroll-s1-1.csv", index_col=0)
    features = ["merit", "SAT", "GPA"]
    target = "enroll"
    
    regression = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=1)
    regression.fit(X=historical_data.loc[:, features], y=historical_data.loc[:, target])

    studentsdata_full = pd.read_csv(janos_data_url + "college_applications6000.csv", index_col=0)
    nstudents = 100 
    studentsdata = studentsdata_full.sample(nstudents, random_state=1)

    def create_optimization_problem(m):
        m.Params.TimeLimit = 60 # More time for larger problem
        y_prob = gppd.add_vars(m, studentsdata, name='enroll_probability', lb=0, ub=1)
        # Merit variables
        merit_vars = gppd.add_vars(m, studentsdata, lb=0.0, ub=2.5, name='merit')
        
        # Build input dataframe for predictor
        # We need the full feature set (merit, SAT, GPA)
        # merit is variable, others are fixed from studentsdata
        input_data = studentsdata.copy()
        input_data['merit'] = merit_vars
        input_data = input_data[features]
        
        m.setObjective(y_prob.sum(), gp.GRB.MAXIMIZE)
        m.addConstr(merit_vars.sum() <= 0.2 * nstudents)
        
        return input_data, y_prob, {"epsilon": 1e-5}

    return regression, create_optimization_problem

# Main Execution
if __name__ == "__main__":
    import sys
    verbose = "-v" in sys.argv
    all_results = []
    
    # Avocado Case
    rf_reg, avocado_problem = setup_avocado()
    all_results.extend(benchmark_formulations("Avocado", rf_reg, avocado_problem, verbose=verbose))
    
    # Janos Case
    janos_reg, janos_problem = setup_janos()
    all_results.extend(benchmark_formulations("Janos", janos_reg, janos_problem, verbose=verbose))
    
    print("\nBenchmark Complete.")
