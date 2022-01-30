"""
@Author: Caleb Ju

Periodic (T=1+1) inventory/newsvendor problem
"""
from msppy.msp import MSLP
from msppy.solver import PSDDP
import gurobipy

import numpy as np

MINIMIZATION =  1
MAXIMIZATION = -1
INF = float('inf')

# Costs 
T = 2
T_inf = 100
[c,b,h] = [2.0,2.8,0.2]
N = 1
discount = 0.8

# Uncertainity and initial conditions
d = 9.0
phi = 0.6
x_0 = 10
D_0 = 5.5
np.random.seed(1)
demand_scenario = d*np.ones(N) + phi*np.random.uniform(size=N)

inventory = MSLP(T=T, sense=MINIMIZATION, bound=0, discount=discount)

for t in range(T):
    m = inventory[t]

    now, past = m.addStateVar(lb=-INF, ub=INF, name="stock")
    u = m.addVars(3, obj=[c,b,h], lb=np.zeros(3), name="control")
    demand = m.addVar(name="demand")

    if t > 0:
        m.addConstr(demand == 0, uncertainty={'rhs': demand_scenario})

    if t == 0:
        m.addConstr(demand == D_0)
        m.addConstr(past == x_0)

    m.addConstr(now - (past + u[0]) == -demand, name="balance")
    m.addConstr(u[1] + (past + u[0]) >= demand, name="undersupply")
    m.addConstr(u[2] - (past + u[0]) >= -demand, name="oversupply")

HT_psddp = PSDDP(inventory)
HT_psddp.solve(max_iterations=110, forward_T=100)

avg = np.mean(demand_scenario)
opt = (4.5*0.2) + discount*((avg-4.5)*2) + discount**2/(1-discount)*avg*2
print("OPT:", opt)
