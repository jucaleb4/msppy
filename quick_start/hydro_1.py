"""
@Author: Caleb Ju

Periodical code adapated from: https://github.com/lingquant/msppy/blob/master/doc/source/examples/hydro_thermal/infinity.ipynb
Opt and uncertainity adapted from: https://github.com/lingquant/msppy/blob/master/doc/source/examples/hydro_thermal/introduction.ipynb
"""
import pandas
import numpy as np
import numpy.linalg as la
import gurobipy
from msppy.msp import MSLP
from msppy.solver import PSDDP
from msppy.evaluation import EvaluationTrue, Evaluation
import sys

np.random.seed(1)
fname = "../doc/source/examples/hydro_thermal/data/"
discount = 0.9906
N = 10

hydro_ = pandas.read_csv(fname + "hydro.csv", index_col=0)
demand = pandas.read_csv(fname + "demand.csv", index_col=0)
deficit_ = pandas.read_csv(fname + "deficit.csv", index_col=0)
exchange_ub = pandas.read_csv(fname + "exchange.csv", index_col=0)
exchange_cost = pandas.read_csv(fname + "exchange_cost.csv", index_col=0)
thermal_ = [pandas.read_csv(fname + "thermal_{}.csv".format(i),
    index_col=0) for i in range(4)]

# historical rainfall data
hist = [pandas.read_csv(fname + "hist_{}.csv".format(i), sep=";") for i in range(4)]
hist = pandas.concat(hist, axis=1)
hist.dropna(inplace=True)
hist.drop(columns='YEAR', inplace=True)
scenarios = [hist.iloc[:,12*i:12*(i+1)].transpose().values for i in range(4)]
# [region][month][year]
scenarios = np.array(scenarios)
scenarios = np.mean(scenarios, axis=1)

nr = 4
assert nr == scenarios.shape[0]

means  = np.mean(scenarios, axis=1)
sigmas = np.std(scenarios, axis=1)

lognorm_sigmas= np.sqrt(np.log(np.power(np.divide(sigmas,means), 2) + 1))
lognorm_means = np.log(means) - np.square(lognorm_sigmas)/2

scenarios = np.array([np.random.lognormal(
                      mean =lognorm_means[i], 
                      sigma=lognorm_sigmas[i], size=N) 
                      for i in range(nr)])
scenario_0 = np.array([hydro_['INITIAL'][nr:2*nr].to_numpy()]).T

HydroThermal = MSLP(T=2, bound=0, discount=discount)

demand = demand.to_numpy()
# get monthly avg
demand = np.mean(demand, axis=0)

for t in range(2):
    m = HydroThermal[t]
    stored_now, stored_past = m.addStateVars(4, ub=hydro_['UB'][:4], name="stored")
    # stored_now, stored_past = m.addStateVars(4, lb=[50000,5000,10000,5000], ub=[75000,7500,15000,7500], name="stored")
    spill = m.addVars(4, name="spill", obj=0.001)
    hydro = m.addVars(4, ub=hydro_['UB'][-4:], name="hydro")    
    deficit = m.addVars(
        [(i,j) for i in range(4) for j in range(4)], 
        ub = [demand[i] * deficit_['DEPTH'][j] for i in range(4) for j in range(4)],
        obj = [deficit_['OBJ'][j] for i in range(4) for j in range(4)], 
        name = "deficit")
    thermal = [None] * 4
    for i in range(4):
        thermal[i] = m.addVars(
            len(thermal_[i]), 
            ub=thermal_[i]['UB'], 
            lb=thermal_[i]['LB'], 
            obj=thermal_[i]['OBJ'], 
            name="thermal_{}".format(i)
        )
    exchange = m.addVars(5,5, obj=exchange_cost.values.flatten(),
        ub=exchange_ub.values.flatten(), name="exchange")    
    thermal_sum = m.addVars(4, name="thermal_sum")
    m.addConstrs(thermal_sum[i] == gurobipy.quicksum(thermal[i].values()) for i in range(4))
    
    for i in range(4): 
        m.addConstr(
            thermal_sum[i] 
            + gurobipy.quicksum(deficit[(i,j)] for j in range(4)) 
            + hydro[i] 
            - gurobipy.quicksum(exchange[(i,j)] for j in range(5))
            + gurobipy.quicksum(exchange[(j,i)] for j in range(5)) 
            == demand[i]
        )
    m.addConstr(
        gurobipy.quicksum(exchange[(j,4)] for j in range(5)) 
        - gurobipy.quicksum(exchange[(4,j)] for j in range(5)) 
        == 0
    )
    for i in range(4):
        if t == 0:
            m.addConstr(
                stored_now[i] + spill[i] + hydro[i] - stored_past[i] 
                == scenario_0[i]
            )
        else:
            m.addConstr(
                stored_now[i] + spill[i] + hydro[i] - stored_past[i] == 0, 
                uncertainty={'rhs': scenarios[i]}
            )
    if t == 0:
        m.addConstrs(stored_past[i] == hydro_['INITIAL'][:4][i] for i in range(4))

HT_psddp = PSDDP(HydroThermal)
HT_psddp.solve(max_iterations=10000, forward_T=60)

# result = Evaluation(HydroThermal)
# result.run(n_simulations=200, query_T=240)
# print(result.CI)
