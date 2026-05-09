import gurobipy as gp
import gurobipy_pandas as gppd
import numpy as np
import pandas as pd
import time
import cProfile
import pstats
from sklearn.ensemble import RandomForestRegressor

from gurobi_ml import add_predictor_constr

# --- Setup Janos Case (100 students, 50 trees) ---
janos_data_url = "https://raw.githubusercontent.com/INFORMSJoC/2020.1023/master/data/"
historical_data = pd.read_csv(janos_data_url + "college_student_enroll-s1-1.csv", index_col=0)
features = ["merit", "SAT", "GPA"]
target = "enroll"

regression = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=1)
regression.fit(X=historical_data.loc[:, features], y=historical_data.loc[:, target])

studentsdata_full = pd.read_csv(janos_data_url + "college_applications6000.csv", index_col=0)
nstudents = 100
studentsdata = studentsdata_full.sample(nstudents, random_state=1)

def run_build():
    m = gp.Model("Profile_Janos")
    m.Params.OutputFlag = 0
    
    y_prob = gppd.add_vars(m, studentsdata, name='enroll_probability', lb=0, ub=1)
    merit_vars = gppd.add_vars(m, studentsdata, lb=0.0, ub=2.5, name='merit')
    
    input_data = studentsdata.copy()
    input_data['merit'] = merit_vars
    input_data = input_data[features]
    
    m.setObjective(y_prob.sum(), gp.GRB.MAXIMIZE)
    m.addConstr(merit_vars.sum() <= 0.2 * nstudents)
    
    add_predictor_constr(m, regression, input_data, y_prob, formulation="vidal", epsilon=1e-5, name="")

# Profile the build
profiler = cProfile.Profile()
profiler.enable()
run_build()
profiler.disable()

stats = pstats.Stats(profiler).sort_stats('tottime')
stats.print_stats(30)
